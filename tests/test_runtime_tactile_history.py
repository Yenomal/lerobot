"""Runtime history follows accepted observations rather than model scheduling."""

from threading import Event, Lock
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.inference import FeatureSpec, ObservationSnapshot, PolicyRunner
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import NormalizerProcessorStep
from lerobot.runtime_server import RuntimeSession
from lerobot.runtime_workers import ProcessStreamingRunner, _worker_main

KEY = "observation.tactile.left_marker"


def features(value):
    return {KEY: np.full((2, 3, 2), value, dtype=np.float32)}


def runner(streaming=False):
    config = SimpleNamespace(
        streaming=streaming,
        use_fm_tactile_token=True,
        use_vlm_tactile_token=True,
        fm_tactile_token_history_length=3,
        vlm_tactile_token_history_length=5,
        streaming_chunk_size=3,
        chunk_size=6,
    )
    return SimpleNamespace(policy=SimpleNamespace(config=config), reset=lambda **kwargs: None)


def test_queued_actions_still_collect_history_and_replays_do_not_duplicate():
    stub = runner()
    seen = []

    def predict(observation, **kwargs):
        seen.append(observation)
        return SimpleNamespace(canonical_actions=torch.zeros(3, 1))

    stub.predict = predict
    session = RuntimeSession(stub, queue_size=3)
    for step in range(4):
        result = session.infer(features(step), "task", step)
        assert session.infer(features(100), "task", step)["replayed"]
        session.acknowledge(result["action_id"])
    assert len(seen) == 2
    assert seen[0].tactile_history[KEY][:, 0, 0, 0].tolist() == [0] * 5
    assert seen[1].tactile_history[KEY][:, 0, 0, 0].tolist() == [0, 0, 1, 2, 3]
    assert not seen[1].tactile_history[KEY].flags.writeable
    assert list(seen[1].tactile_history_times) == sorted(seen[1].tactile_history_times)
    result = session.infer(features(9), "new task", 4)
    assert seen[-1].tactile_history[KEY][:, 0, 0, 0].tolist() == [9] * 5
    session.acknowledge(result["action_id"])
    session.reset()
    assert not session._tactile_history


def test_slow_prefix_and_fm_share_contiguous_history_with_independent_snapshots():
    stub = runner(streaming=True)
    prefix_seen, fm_seen = [], []
    started, release = Event(), Event()

    def encode(observation):
        prefix_seen.append(observation)
        if len(prefix_seen) == 2:
            started.set()
            assert release.wait(10)
        return {}, {}

    def predict(prepared, cache, *, observation, **kwargs):
        fm_seen.append(observation)
        return SimpleNamespace(canonical_actions=torch.zeros(3, 1))

    stub.encode_prefix, stub.predict_from_prefix = encode, predict
    session = RuntimeSession(stub, queue_size=3, streaming_horizon=4, streaming_watermark=2)
    try:
        for step in range(4):
            result = session.infer(features(step), "task", step)
            session.acknowledge(result["action_id"])
            if step == 2:
                assert started.wait(5)
        assert prefix_seen[1].tactile_history[KEY][:, 0, 0, 0].tolist() == [0, 0, 0, 1, 2]
        assert fm_seen[1].tactile_history[KEY][:, 0, 0, 0].tolist() == [0, 0, 1, 2, 3]
        assert prefix_seen[1].observation_id == "2" and fm_seen[1].observation_id == "3"
    finally:
        release.set()
        session.close()


def test_history_is_normalized_under_original_feature_key():
    instance = object.__new__(PolicyRunner)
    instance.capabilities = SimpleNamespace(
        features=(FeatureSpec(KEY, (2, 3, 2), "float32", semantics="marker coordinates"),)
    )
    instance.max_text_input, instance.robot_type = 20, ""
    raw = np.stack([features(i)[KEY] for i in (1, 3, 5)])
    observation = ObservationSnapshot(features(5), 1, "task", tactile_history={KEY: raw})
    batch = instance._batch(observation)
    assert batch[KEY].shape == (1, 3, 2, 3, 2)
    normalizer = NormalizerProcessorStep(
        features={KEY: PolicyFeature(type=FeatureType.STATE, shape=(2, 3, 2))},
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
        stats={KEY: {"mean": torch.ones(2, 3, 2), "std": torch.full((2, 3, 2), 2.0)}},
    )
    normalized = normalizer({TransitionKey.OBSERVATION: {KEY: batch[KEY]}})[TransitionKey.OBSERVATION][KEY]
    torch.testing.assert_close(normalized[0, :, 0, 0, 0], torch.tensor([0.0, 1.0, 2.0]))
    instance.preprocessor = lambda value: {
        **value,
        KEY: normalizer({TransitionKey.OBSERVATION: {KEY: value[KEY]}})[TransitionKey.OBSERVATION][KEY],
    }
    prepared = instance._prepare_observation(observation)
    assert prepared["_runtime_tactile_history"] is True
    torch.testing.assert_close(prepared[KEY], normalized)
    raw[:] = 999
    assert observation.tactile_history[KEY][0, 0, 0, 0] == 1
    with pytest.raises(ValueError, match="Tactile history shape/dtype/value"):
        instance._batch(
            ObservationSnapshot(
                features(5), 1, "task", tactile_history={KEY: np.zeros((3, 4), dtype=np.float32)}
            )
        )


@pytest.mark.parametrize("role", ["vlm", "fm"])
def test_worker_transport_preserves_tactile_history_and_hmm_state(monkeypatch, role):
    seen, requests = [], []
    proxy = ProcessStreamingRunner.__new__(ProcessStreamingRunner)
    proxy._vlm_lock = Lock()
    proxy._fm_lock = Lock()
    proxy._session_id = "robot-a"
    proxy._fm_prefix_id, proxy._fm_prefix_cache = None, None
    response = (
        {"prepared": {}, "cache": {}} if role == "vlm" else {"actions": np.zeros((1, 1), dtype=np.float32)}
    )
    connection = SimpleNamespace(send=requests.append, recv=lambda: response)
    proxy._vlm_parent = proxy._fm_parent = connection
    snapshot = ObservationSnapshot(
        features(3),
        3,
        "task",
        observation_id="7",
        tactile_history={KEY: np.stack([features(i)[KEY] for i in (1, 2, 3)])},
        tactile_history_times=(1, 2, 3),
        tactile_hmm_state=6,
    )
    if role == "vlm":
        proxy.encode_prefix(snapshot)
    else:
        proxy.predict_from_prefix(
            {}, {}, observation=snapshot, inference_mode="stream", context_cfg_lambda=0.1
        )

    def encode(observation):
        seen.append(observation)
        return {}, {}

    def predict(*args, observation, **kwargs):
        seen.append(observation)
        return SimpleNamespace(canonical_actions=torch.zeros(1, 1))

    monkeypatch.setattr(
        "lerobot.runtime_workers.load_runtime_runner",
        lambda *args, **kwargs: SimpleNamespace(
            for_session=lambda session_id: SimpleNamespace(encode_prefix=encode, predict_from_prefix=predict)
        ),
    )
    incoming = iter([requests[0], {"op": "close"}])
    _worker_main(
        role,
        "checkpoint",
        "cpu",
        SimpleNamespace(send=lambda response: None, recv=lambda: next(incoming), close=lambda: None),
    )
    np.testing.assert_array_equal(seen[0].tactile_history[KEY], snapshot.tactile_history[KEY])
    assert seen[0].tactile_history_times == (1, 2, 3)
    assert seen[0].tactile_hmm_state == 6
    assert seen[0].capture_time == 3

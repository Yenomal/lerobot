from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.inference import ObservationSnapshot
from lerobot.runtime_workers import ProcessStreamingRunner, _decode_cache, _encode_cache, _worker_main


def test_dynamic_cache_codec_round_trips_cpu_tensor_layers():
    from transformers import DynamicCache

    keys = torch.randn(1, 2, 3, 4)
    values = torch.randn(1, 2, 3, 4)
    cache = DynamicCache(ddp_cache_data=((keys, values),))

    encoded = _encode_cache(cache)
    decoded = _decode_cache(encoded, "cpu")

    assert len(decoded.layers) == 1
    torch.testing.assert_close(decoded.layers[0].keys, keys)
    torch.testing.assert_close(decoded.layers[0].values, values)


@pytest.mark.parametrize("unit", ["tokens", "chunks"])
@pytest.mark.parametrize("mode", ["stream-head", "stream-head-mean-tactile-cfg"])
def test_process_proxy_and_fm_worker_forward_tactile_guidance_and_delay(monkeypatch, unit, mode):
    sent = []
    actions = np.zeros((2, 3), dtype=np.float32)
    proxy = ProcessStreamingRunner.__new__(ProcessStreamingRunner)
    proxy._fm_prefix_cache = None
    proxy._fm_prefix_id = None
    proxy._fm_parent = SimpleNamespace(send=sent.append, recv=lambda: {"actions": actions})
    observation = ObservationSnapshot({}, 0.0, "task", observation_id="1")
    proxy.predict_from_prefix(
        {},
        {},
        observation=observation,
        inference_mode=mode,
        context_cfg_lambda=0.1,
        tactile_cfg_lambda=2.25,
        **{f"observation_delay_{unit}": 3},
    )
    seen = {}

    def predict_from_prefix(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(canonical_actions=torch.from_numpy(actions))

    runner = SimpleNamespace(predict_from_prefix=predict_from_prefix)
    monkeypatch.setattr("lerobot.runtime_workers.load_runtime_runner", lambda *args: runner)
    requests = iter([sent[0], {"op": "close"}])
    replies = []
    connection = SimpleNamespace(send=replies.append, recv=lambda: next(requests), close=lambda: None)
    _worker_main("fm", "checkpoint", "cpu", connection)
    assert seen["inference_mode"] == mode
    assert seen["tactile_cfg_lambda"] == 2.25
    assert seen[f"observation_delay_{unit}"] == 3
    np.testing.assert_array_equal(replies[1]["actions"], actions)


def test_proxy_submits_prefix_once_until_refresh_or_reset():
    sent = []
    proxy = ProcessStreamingRunner.__new__(ProcessStreamingRunner)
    proxy._vlm_lock = Lock()
    proxy._fm_prefix_cache = None
    proxy._fm_prefix_id = None
    proxy._vlm_parent = SimpleNamespace(send=lambda request: None, recv=lambda: {"ok": True})
    proxy._fm_parent = SimpleNamespace(
        send=sent.append, recv=lambda: {"actions": np.zeros((1, 1), dtype=np.float32)}
    )
    first = {"prefix_pad_masks": torch.ones(1, 2, dtype=torch.bool)}
    refreshed = {"prefix_pad_masks": torch.ones(1, 3, dtype=torch.bool)}

    def predict(cache, seq):
        proxy.predict_from_prefix(
            {"unused": torch.ones(5)},
            cache,
            observation=ObservationSnapshot({"state": np.array([seq])}, 0.0, "task", observation_id=str(seq)),
            inference_mode="stream-head",
            context_cfg_lambda=0.1,
            observation_delay_tokens=seq,
        )

    predict(first, 0)
    predict(first, 1)
    predict(refreshed, 2)
    predict(refreshed, 3)
    assert "cache" in sent[0] and "cache" not in sent[1]
    assert "cache" in sent[2] and "cache" not in sent[3]
    assert sent[0]["cache_id"] == sent[1]["cache_id"]
    assert sent[2]["cache_id"] == sent[3]["cache_id"] != sent[0]["cache_id"]
    assert all("prepared" not in request for request in sent)
    assert [request["observation_delay_tokens"] for request in sent] == [0, 1, 2, 3]
    assert [request["features"]["state"].item() for request in sent] == [0, 1, 2, 3]
    proxy.reset()
    assert proxy._fm_prefix_cache is None and proxy._fm_prefix_id is None
    predict(refreshed, 4)
    assert sent[-1]["cache"] is refreshed
    assert sent[-1]["cache_id"] != sent[2]["cache_id"]


def test_fm_worker_reuses_and_replaces_device_cache_and_clears_it_on_reset(monkeypatch):
    from transformers import DynamicCache

    def payload(value):
        keys = torch.full((1, 1, 2, 2), value)
        return {
            "prefix_pad_masks": torch.ones(1, 2, dtype=torch.bool),
            "past_key_values": _encode_cache(DynamicCache(ddp_cache_data=((keys, keys.clone()),))),
        }

    decoded = []
    original_decode = _decode_cache

    def decode(value, device):
        if isinstance(value, dict) and "past_key_values" in value:
            decoded.append(value)
        return original_decode(value, device)

    monkeypatch.setattr("lerobot.runtime_workers._decode_cache", decode)
    retained = []
    observed = []
    resets = []

    def predict(prepared, cache, **kwargs):
        assert prepared == {}
        retained.append(cache)
        kv = cache["past_key_values"]
        observed.append(kv.layers[0].keys.clone())
        # Mirror SmolVLA's append/crop lifecycle on the persistent DynamicCache.
        suffix = torch.full((1, 1, 1, 2), 99.0)
        kv.update(suffix, suffix, 0)
        kv.layers[0].crop(2 - kv.layers[0].get_seq_length())
        return SimpleNamespace(canonical_actions=observed[-1].reshape(2, 2))

    runner = SimpleNamespace(predict_from_prefix=predict, reset=lambda **kwargs: resets.append(True))
    monkeypatch.setattr("lerobot.runtime_workers.load_runtime_runner", lambda *args: runner)
    first, second = payload(1.0), payload(2.0)

    def request(cache_id, cache=None):
        result = {
            "op": "predict_from_prefix",
            "cache_id": cache_id,
            "features": {},
            "task": "task",
            "sequence": 1,
            "inference_mode": "stream-head",
            "context_cfg_lambda": 0.1,
            "tactile_cfg_lambda": 1.5,
        }
        if cache is not None:
            result["cache"] = cache
        return result

    requests = iter(
        [
            request("first", first),
            request("first"),
            request("second", second),
            request("second"),
            {"op": "reset"},
            request("third", first),
            {"op": "close"},
        ]
    )
    replies = []
    connection = SimpleNamespace(send=replies.append, recv=lambda: next(requests), close=lambda: None)
    _worker_main("fm", "checkpoint", "cpu", connection)
    assert len(decoded) == 3
    assert retained[0] is retained[1] and retained[2] is retained[3]
    assert retained[0] is not retained[2] and retained[0] is not retained[4]
    assert len(resets) == 1
    for actual, expected in zip(observed, [1.0, 1.0, 2.0, 2.0, 1.0], strict=True):
        torch.testing.assert_close(actual, torch.full_like(actual, expected))


@pytest.mark.parametrize("scenario", ["missing", "mismatch", "reset"])
def test_fm_worker_rejects_cache_reference_without_matching_loaded_prefix(monkeypatch, scenario):
    runner = SimpleNamespace(
        reset=lambda **kwargs: None,
        predict_from_prefix=lambda *args, **kwargs: SimpleNamespace(canonical_actions=torch.zeros(1, 1)),
    )
    monkeypatch.setattr("lerobot.runtime_workers.load_runtime_runner", lambda *args: runner)
    initial = {
        "op": "predict_from_prefix",
        "cache_id": "loaded",
        "cache": {},
        "features": {},
        "task": "task",
        "sequence": 0,
        "inference_mode": "stream-head",
        "context_cfg_lambda": 0.1,
        "tactile_cfg_lambda": 1.5,
    }
    requests = iter(
        ([initial] if scenario != "missing" else [])
        + ([{"op": "reset"}] if scenario == "reset" else [])
        + [{"op": "predict_from_prefix", "cache_id": "loaded" if scenario == "reset" else "missing"}]
    )
    replies = []
    connection = SimpleNamespace(send=replies.append, recv=lambda: next(requests), close=lambda: None)
    with pytest.raises(ValueError, match="FM prefix cache is not loaded"):
        _worker_main("fm", "checkpoint", "cpu", connection)
    assert "error" in replies[-1]


def test_reset_serializes_with_pending_vlm_refresh(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    proxy = ProcessStreamingRunner.__new__(ProcessStreamingRunner)
    proxy._vlm_lock = Lock()
    proxy._fm_prefix_id = "old"
    proxy._fm_prefix_cache = {}
    received = Event()
    release = Event()
    resetting = Event()
    operations = []

    def send(request):
        operations.append(request["op"])

    def recv():
        if operations[-1] == "encode_prefix":
            received.set()
            assert release.wait(timeout=5)
            return {"prepared": {}, "cache": {"prefix": "new"}}
        return {"ok": True}

    proxy._vlm_parent = SimpleNamespace(send=send, recv=recv)
    proxy._fm_parent = SimpleNamespace(send=lambda request: None, recv=lambda: {"ok": True})

    def reset():
        resetting.set()
        proxy.reset()

    with ThreadPoolExecutor(max_workers=2) as executor:
        refresh = executor.submit(proxy.encode_prefix, ObservationSnapshot({}, 0.0, "task"))
        assert received.wait(timeout=5)
        pending_reset = executor.submit(reset)
        try:
            assert resetting.wait(timeout=5)
            assert operations == ["encode_prefix"]
        finally:
            release.set()
        assert refresh.result(timeout=5) == ({}, {"prefix": "new"})
        pending_reset.result(timeout=5)
    assert operations == ["encode_prefix", "reset"]
    assert proxy._fm_prefix_id is None and proxy._fm_prefix_cache is None

"""Physical marker displacement, independent of absolute-coordinate normalization."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.policies.smolvla.processor_smolvla import configure_tactile_delta_preprocessor
from lerobot.processor import NormalizerProcessorStep
from tests.policies.smolvla import test_delay_adarms

make_model = test_delay_adarms.make_model


def marker_features():
    return {
        f"observation.tactile.{side}_marker": PolicyFeature(type=FeatureType.STATE, shape=(2, 63, 2))
        for side in ("left", "right")
    }


@pytest.mark.parametrize("tcn", [False, True])
def test_delta_encoder_widths_values_and_training_cached_parity(make_model, tcn):
    torch.manual_seed(63)
    model = VLAFlowMatching(
        replace(
            make_model(tactile=True, enabled=False).config,
            use_tactile_delta=True,
            use_vlm_tactile_token=True,
            use_tactile_tcn=tcn,
            tactile_token_history_length=(5 if tcn else 3),
            input_features=marker_features(),
        )
    )
    batch = {key: torch.randn(2, 5 if tcn else 3, 2, 63, 2) for key in marker_features()}
    captured = []
    handle = model.tactile_encoder.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    model._encode_tactile(batch)
    handle.remove()
    expected = torch.cat(
        [value[:, :, 1].flatten(2) - value[:, :, 0].flatten(2) for value in batch.values()], -1
    )
    torch.testing.assert_close(captured[0], expected.reshape(-1, 252))
    model._encode_tactile(batch, vlm=True)
    assert model.tactile_encoder[0].in_features == model.vlm_tactile_encoder[0].in_features == 252
    shifted = {key: value + torch.randn(2, 5 if tcn else 3, 1, 63, 2) for key, value in batch.items()}
    torch.testing.assert_close(model._encode_tactile(batch), model._encode_tactile(shifted))
    policy = object.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config, policy.model = model.config, model
    # A fresh model initialized from metadata must have the same input width.
    policy.model = VLAFlowMatching(model.config)
    assert policy.initialize_lazy_parameters()
    assert (
        policy.model.tactile_encoder[0].in_features == policy.model.vlm_tactile_encoder[0].in_features == 252
    )
    lang, valid, state = torch.tensor([[1, 2], [3, 4]]), torch.ones(2, 2, dtype=torch.bool), torch.randn(2, 6)
    actions, noise, time = torch.randn(2, 4, 6), torch.randn(2, 4, 6), torch.full((2, 4), 0.5)
    cache = model.encode_prefix([], [], lang, valid, state, tactile_batch=batch)
    raw, ca0 = model.denoise_step(
        cache["prefix_pad_masks"],
        cache["past_key_values"],
        0.5 * (actions + noise),
        time,
        tactile_batch=batch,
        return_tactile_copy=True,
        apply_head_guidance=False,
    )
    loss = model([], [], lang, valid, state, actions, noise, time, tactile_batch=batch)
    expected_loss = (raw - (noise - actions)).square()
    expected_loss[:, 0] = 0.5 * (expected_loss[:, 0] + (ca0 - (noise - actions)[:, 0]).square())
    torch.testing.assert_close(loss, expected_loss)
    loss.mean().backward()
    assert model.tactile_encoder[0].weight.grad.abs().sum() > 0


def test_saved_processor_stats_do_not_distort_delta(monkeypatch):
    features = {**marker_features(), "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))}
    stats = {
        key: {"mean": torch.full(feature.shape, 10.0), "std": torch.full(feature.shape, 3.0)}
        for key, feature in features.items()
    }
    normalizer = NormalizerProcessorStep(
        features=features, norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD}, stats=stats
    )
    pipeline = SimpleNamespace(steps=[normalizer])
    config = SmolVLAConfig(use_tactile=True, use_fm_tactile_token=True, use_tactile_delta=True)
    monkeypatch.setattr(
        "lerobot.policies.factory._make_pretrained_processors_from_policy_config", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "lerobot.policies.factory.load_pretrained_policy_processors", lambda *args, **kwargs: (pipeline, None)
    )
    make_pre_post_processors(config, pretrained_path="mock-checkpoint")
    raw = {key: torch.randn(1, 5, *feature.shape) for key, feature in features.items()}
    result = normalizer({TransitionKey.OBSERVATION: raw})[TransitionKey.OBSERVATION]
    for key in marker_features():
        torch.testing.assert_close(result[key], raw[key], rtol=0, atol=0)
    torch.testing.assert_close(
        result["observation.state"], (raw["observation.state"] - 10) / (3 + normalizer.eps)
    )
    assert normalizer.get_config()["normalize_observation_keys"] == ["observation.state"]
    configure_tactile_delta_preprocessor(config, pipeline)
    assert normalizer.normalize_observation_keys == {"observation.state"}


def test_delta_reconstruction_target_and_legacy_config(make_model, tmp_path):
    config = replace(
        make_model(tactile=True, enabled=False).config,
        use_tactile_delta=True,
        use_tactile_reconstruction=True,
        input_features=marker_features(),
    )
    model = VLAFlowMatching(config)
    batch = {key: torch.randn(2, 1, 2, 63, 2) for key in marker_features()}
    embedding = model._encode_tactile(batch)[:, -1]
    target = torch.cat(
        [value[:, -1, 1].flatten(1) - value[:, -1, 0].flatten(1) for value in batch.values()], -1
    )
    assert model.tactile_reconstruction[-1].out_features == 252
    lang = torch.ones(2, 2, dtype=torch.long)
    losses = model([], [], lang, lang.bool(), torch.randn(2, 6), torch.randn(2, 4, 6), tactile_batch=batch)
    torch.testing.assert_close(losses[:, 0], (model.tactile_reconstruction(embedding) - target).square())
    assert not SmolVLAConfig().use_tactile_delta
    config.save_pretrained(tmp_path)
    assert SmolVLAConfig.from_pretrained(tmp_path).use_tactile_delta
    with pytest.raises(ValueError, match="use_tactile_delta requires"):
        SmolVLAConfig(use_tactile_delta=True)
    with pytest.raises(ValueError, match="marker shape"):
        model._tactile_marker_values(torch.randn(2, 3, 63, 2))

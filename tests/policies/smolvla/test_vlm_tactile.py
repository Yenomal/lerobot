"""Independent VLM tactile encoding and state-isolated multimodal prefix attention."""

from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from tests.policies.smolvla import test_delay_adarms

make_model = test_delay_adarms.make_model


def inputs():
    return {f"observation.tactile.{side}_marker": torch.randn(2, 3, 2, 3, 2) for side in ("left", "right")}


@pytest.mark.parametrize("fm", [False, True])
@pytest.mark.parametrize("tcn,causal", [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize("no_vlm", [False, True])
def test_independent_encoders_prefix_mask_cache_parity_and_gradients(make_model, fm, tcn, causal, no_vlm):
    torch.manual_seed(43)
    template = make_model(
        enabled=False, tactile=True, expert_width_multiplier=0.5, tactile_cross_attention=False
    )
    model = VLAFlowMatching(
        replace(
            template.config,
            use_vlm_tactile_token=True,
            use_tactile_token=fm,
            use_tactile_token_history=tcn,
            use_tactile_causal_attention=causal,
            tactile_token_history_length=3,
            tactile_no_vlm_attention=no_vlm,
        )
    )
    for layer in model.vlm_with_expert.lm_expert.layers:
        nn.init.normal_(layer.self_attn.o_proj.weight, std=0.5)
    lang, state = torch.tensor([[1, 2], [3, 4]]), torch.randn(2, 6)
    valid = torch.tensor([[True, False], [True, True]])
    batch = inputs()
    embs, pad, roles = model.embed_prefix([], [], lang, valid, state, tactile_batch=batch)
    count = 1 if tcn else 3
    assert embs.shape == (2, 3 + count, 8)
    assert model.vlm_tactile_encoder[0].out_features == model.vlm_tactile_encoder[2].out_features == 4
    assert model.vlm_tactile_projector.in_features == 4 and model.vlm_tactile_projector.out_features == 8
    prefix_mask = model._apply_vlm_tactile_prefix_mask(make_att_2d_masks(pad, roles), pad, roles)
    assert not prefix_mask[:, 3:, 2].any()  # T never reads state.
    assert not prefix_mask[:, :2, 2].any()  # Image/language never read state.
    torch.testing.assert_close(prefix_mask[:, :2, 3:], valid[:, :, None].expand(-1, -1, count))
    expected = torch.ones(2, count, count, dtype=torch.bool)
    if causal:
        expected = expected.tril()
    torch.testing.assert_close(prefix_mask[:, 3:, 3:], expected)
    if no_vlm:
        assert not prefix_mask[:, 3:, :2].any()
    else:
        torch.testing.assert_close(prefix_mask[:, 3:, :2], valid[:, None].expand(-1, count, -1))
    first = model.vlm_with_expert(
        inputs_embeds=[embs, None],
        attention_mask=prefix_mask,
        position_ids=pad.long().cumsum(-1) - 1,
        use_cache=True,
    )[0][0]
    other_embs, _, _ = model.embed_prefix([], [], lang, valid, -20 * state, tactile_batch=batch)
    other = model.vlm_with_expert(
        inputs_embeds=[other_embs, None],
        attention_mask=prefix_mask,
        position_ids=pad.long().cumsum(-1) - 1,
        use_cache=True,
    )[0][0]
    torch.testing.assert_close(first[:, 3:], other[:, 3:], rtol=0, atol=0)
    changed_embs, _, _ = model.embed_prefix(
        [], [], lang, valid, state, tactile_batch={k: -10 * v for k, v in batch.items()}
    )
    changed = model.vlm_with_expert(
        inputs_embeds=[changed_embs, None],
        attention_mask=prefix_mask,
        position_ids=pad.long().cumsum(-1) - 1,
        use_cache=True,
    )[0][0]
    assert not torch.allclose(first[:, :1], changed[:, :1])  # Graph is bidirectional with language.
    cache = model.encode_prefix([], [], lang, valid, state, tactile_batch=batch)
    saved_keys = [layer.keys.clone() for layer in cache["past_key_values"].layers]
    actions, noise, time = torch.randn(2, 4, 6), torch.randn(2, 4, 6), torch.full((2, 4), 0.5)
    x_t = time[..., None] * noise + (1 - time[..., None]) * actions
    result = model.denoise_step(
        cache["prefix_pad_masks"],
        cache["past_key_values"],
        x_t,
        time,
        tactile_batch=batch,
        return_tactile_copy=fm,
        apply_head_guidance=False,
    )
    raw, copy = result if fm else (result, None)
    loss = model([], [], lang, valid, state, actions, noise, time, tactile_batch=batch)
    expected_loss = (raw - (noise - actions)).square()
    if fm:
        expected_loss[:, 0] = 0.5 * (expected_loss[:, 0] + (copy - (noise - actions)[:, 0]).square())
    torch.testing.assert_close(loss, expected_loss)
    for saved, layer in zip(saved_keys, cache["past_key_values"].layers, strict=True):
        torch.testing.assert_close(saved, layer.keys)
    loss.mean().backward()
    assert model.vlm_tactile_encoder[0].weight.grad.abs().sum() > 0
    assert model.vlm_tactile_projector.weight.grad.abs().sum() > 0
    if tcn:
        assert model.vlm_tactile_token_tcn_1.weight.grad.abs().sum() > 0
    if fm:
        for vlm_parameter, fm_parameter in zip(
            model.vlm_tactile_encoder.parameters(), model.tactile_encoder.parameters(), strict=True
        ):
            assert vlm_parameter is not fm_parameter
            assert vlm_parameter.data_ptr() != fm_parameter.data_ptr()
        assert model.tactile_encoder[0].weight.grad.abs().sum() > 0
        if tcn:
            assert model.vlm_tactile_token_tcn_in.weight is not model.tactile_token_tcn_in.weight


def test_lazy_initialization_checkpoint_and_config(make_model, tmp_path):
    template = make_model(tactile=True, expert_width_multiplier=0.5)
    config = replace(
        template.config,
        use_vlm_tactile_token=True,
        use_tactile_token=False,
        use_tactile_token_history=True,
        tactile_token_history_length=3,
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        **{
            f"observation.tactile.{side}_marker": PolicyFeature(type=FeatureType.STATE, shape=(2, 3, 2))
            for side in ("left", "right")
        },
    }
    model = VLAFlowMatching(config)
    policy = object.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config, policy.model = config, model
    assert policy.initialize_lazy_parameters()
    assert not policy.initialize_lazy_parameters()
    assert model.vlm_tactile_encoder[0].in_features == model.tactile_encoder[0].in_features == 24
    file = str(tmp_path / "model.safetensors")
    save_file(policy.state_dict(), file)
    SmolVLAPolicy._load_as_safetensor(policy, file, "cpu", True)
    config.save_pretrained(tmp_path)
    loaded = SmolVLAConfig.from_pretrained(tmp_path)
    assert loaded.use_vlm_tactile_token and not loaded.use_tactile_token
    assert loaded.use_tactile_token_history
    assert "vlm_tactile_projector" in policy._get_default_peft_targets()["target_modules"]
    with pytest.raises(ValueError, match="use_vlm_tactile_token requires"):
        SmolVLAConfig(use_vlm_tactile_token=True)


def test_vlm_and_fm_runtime_history_and_delayed_training_frames(make_model):
    template = make_model(tactile=True)
    model = VLAFlowMatching(
        replace(template.config, use_vlm_tactile_token=True, tactile_token_history_length=3)
    )
    policy = object.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config, policy.model = model.config, model
    policy.prepare_images = lambda batch: ([], [])
    policy.reset()
    policy.eval()
    lang = torch.tensor([[1, 2], [3, 4]])
    for step in range(3):
        batch = {
            **{k: torch.full_like(v[:, -1], float(step)) for k, v in inputs().items()},
            "observation.state": torch.zeros(2, 6),
            OBS_LANGUAGE_TOKENS: lang,
            OBS_LANGUAGE_ATTENTION_MASK: torch.ones_like(lang, dtype=torch.bool),
        }
        cache = policy.encode_prefix(dict(batch))
        assert not policy._tactile_history
        history = torch.stack(tuple(policy._vlm_tactile_history["observation.tactile.left_marker"]), dim=1)
        assert history[0, :, 0, 0, 0].tolist() == [float(max(0, step - d)) for d in (2, 1, 0)]
        assert cache["prefix_pad_masks"].shape[1] == 6
    policy.reset()
    assert not policy._vlm_tactile_history and not policy._tactile_history
    batch = {k: torch.arange(7.0).view(1, 7, 1, 1, 1).expand(2, 7, 2, 3, 2) for k in inputs()}
    batch["observation_delay_tokens"] = torch.tensor([2, 0])
    current = model._tactile_frame_values(batch)[0]
    delayed = model._tactile_frame_values(batch, vlm=True)[0]
    assert current[:, :, 0, 0, 0].tolist() == [[4, 5, 6], [4, 5, 6]]
    assert delayed[:, :, 0, 0, 0].tolist() == [[2, 3, 4], [4, 5, 6]]
    assert len(model.config.tactile_observation_delta_indices) == 7


def test_image_language_tactile_prefix_padding_and_position_order(make_model, monkeypatch):
    template = make_model(tactile=True)
    model = VLAFlowMatching(
        replace(template.config, use_vlm_tactile_token=True, tactile_token_history_length=3, prefix_length=12)
    )
    image_tokens = torch.randn(2, 2, 8)
    monkeypatch.setattr(model.vlm_with_expert, "embed_image", lambda image: image_tokens)
    embs, pad, roles = model.embed_prefix(
        [torch.zeros(2, 3, 4, 4)],
        [torch.tensor([True, False])],
        torch.tensor([[1, 2], [3, 4]]),
        torch.ones(2, 2, dtype=torch.bool),
        torch.randn(2, 6),
        tactile_batch=inputs(),
    )
    assert embs.shape == (2, 12, 8)
    assert roles[0].tolist() == [0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0]
    mask = model._apply_vlm_tactile_prefix_mask(make_att_2d_masks(pad, roles), pad, roles)
    assert mask[0, :4, 5:8].all() and mask[0, 5:8, :4].all()
    assert not mask[:, 5:8, 4].any()
    assert not mask[1, :, :2].any() and not mask[1, :2].any()
    assert not mask[:, :, 8:].any() and not mask[:, 8:].any()
    positions = pad.long().cumsum(-1) - 1
    assert positions[0, 4:8].tolist() == [4, 5, 6, 7]


@pytest.mark.parametrize("fm", [False, True])
@pytest.mark.parametrize("tcn", [False, True])
def test_vlm_tactile_bf16_autocast_forward_and_backward(make_model, fm, tcn):
    template = make_model(enabled=False, tactile=True)
    model = VLAFlowMatching(
        replace(
            template.config,
            use_vlm_tactile_token=True,
            use_tactile_token=fm,
            use_tactile_token_history=tcn,
            tactile_token_history_length=3,
        )
    )
    lang = torch.tensor([[1, 2], [3, 4]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(
            [],
            [],
            lang,
            torch.ones_like(lang, dtype=torch.bool),
            torch.randn(2, 6),
            torch.randn(2, 4, 6),
            tactile_batch=inputs(),
        ).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert model.vlm_tactile_projector.weight.grad.abs().sum() > 0
    assert model.vlm_tactile_encoder[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("fm_length,vlm_length", [(2, 5), (7, 2)])
@pytest.mark.parametrize("tcn", [False, True])
def test_separate_lengths_training_delay_and_cached_parity(make_model, fm_length, vlm_length, tcn):
    template = make_model(enabled=False, tactile=True)
    config = replace(
        template.config,
        use_vlm_tactile_token=True,
        use_tactile_token_history=tcn,
        fm_tactile_token_history_length=fm_length,
        vlm_tactile_token_history_length=vlm_length,
    )
    model = VLAFlowMatching(config)
    window = max(fm_length, vlm_length + 4)
    batch = {
        k: torch.arange(float(window)).reshape(1, window, 1, 1, 1).expand(2, window, 2, 3, 2)
        for k in inputs()
    }
    batch["observation_delay_tokens"] = torch.tensor([2, 0])
    fm_frames = model._tactile_frame_values(batch)[0]
    vlm_frames = model._tactile_frame_values(batch, vlm=True)[0]
    assert fm_frames[0, :, 0, 0, 0].tolist() == list(range(window - fm_length, window))
    assert vlm_frames[0, :, 0, 0, 0].tolist() == list(range(window - 2 - vlm_length, window - 2))
    assert vlm_frames[1, :, 0, 0, 0].tolist() == list(range(window - vlm_length, window))
    assert len(config.tactile_observation_delta_indices) == window
    fm_embedding, vlm_embedding = model._encode_tactile(batch), model._encode_tactile(batch, vlm=True)
    assert model._tactile_token_count(fm_embedding) == (1 if tcn else fm_length)
    assert (1 if vlm_embedding.ndim == 2 else vlm_embedding.shape[1]) == (1 if tcn else vlm_length)
    lang, state = torch.tensor([[1, 2], [3, 4]]), torch.randn(2, 6)
    valid = torch.ones_like(lang, dtype=torch.bool)
    cache = model.encode_prefix([], [], lang, valid, state, tactile_batch=batch)
    assert cache["prefix_pad_masks"].shape[1] == 3 + (1 if tcn else vlm_length)
    actions, noise, time = torch.randn(2, 4, 6), torch.randn(2, 4, 6), torch.full((2, 4), 0.5)
    raw, ca0 = model.denoise_step(
        cache["prefix_pad_masks"],
        cache["past_key_values"],
        0.5 * (actions + noise),
        time,
        tactile_batch=batch,
        return_tactile_copy=True,
        apply_head_guidance=False,
    )
    losses = model([], [], lang, valid, state, actions, noise, time, tactile_batch=batch)
    expected = (raw - (noise - actions)).square()
    expected[:, 0] = 0.5 * (expected[:, 0] + (ca0 - (noise - actions)[:, 0]).square())
    torch.testing.assert_close(losses, expected)


@pytest.mark.parametrize("fm_length,vlm_length", [(2, 5), (7, 2)])
def test_separate_runtime_queues_for_full_and_split_paths(make_model, monkeypatch, fm_length, vlm_length):
    template = make_model(enabled=False, tactile=True)
    model = VLAFlowMatching(
        replace(
            template.config,
            use_vlm_tactile_token=True,
            fm_tactile_token_history_length=fm_length,
            vlm_tactile_token_history_length=vlm_length,
        )
    )
    model.config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    policy = object.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config, policy.model = model.config, model
    policy.prepare_images = lambda batch: ([], [])
    policy.reset()
    captures = []

    def sample(*args, **kwargs):
        tactile = kwargs["tactile_batch"]
        fm, vlm = model._tactile_frame_values(tactile)[0], model._tactile_frame_values(tactile, vlm=True)[0]
        captures.append((fm, vlm))
        return torch.zeros(2, 4, 6)

    monkeypatch.setattr(model, "sample_actions", sample)
    for step in range(max(fm_length, vlm_length) + 1):
        batch = {k: torch.full((2, 2, 3, 2), float(step)) for k in inputs()}
        batch.update(
            {
                "observation.state": torch.zeros(2, 6),
                OBS_LANGUAGE_TOKENS: torch.ones(2, 2, dtype=torch.long),
                OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 2, dtype=torch.bool),
            }
        )
        policy.predict_action_chunk(dict(batch))
        fm_frames, vlm_frames = captures[-1]
        assert fm_frames.shape[1] == fm_length and vlm_frames.shape[1] == vlm_length
        assert fm_frames[0, :, 0, 0, 0].tolist() == [
            float(max(0, step - lag)) for lag in reversed(range(fm_length))
        ]
        assert vlm_frames[0, :, 0, 0, 0].tolist() == [
            float(max(0, step - lag)) for lag in reversed(range(vlm_length))
        ]
    policy.reset()
    policy.encode_prefix(dict(batch))
    assert not policy._tactile_history
    assert len(policy._vlm_tactile_history["observation.tactile.left_marker"]) == vlm_length
    policy._prepare_batch(dict(batch))
    assert len(policy._tactile_history["observation.tactile.left_marker"]) == fm_length
    policy.reset()
    assert not policy._tactile_history and not policy._vlm_tactile_history


def test_history_length_config_legacy_fallback_validation_and_round_trip(tmp_path):
    legacy = SmolVLAConfig(tactile_token_history_length=7)
    assert legacy.get_tactile_history_length() == legacy.get_tactile_history_length(vlm=True) == 7
    config = replace(legacy, fm_tactile_token_history_length=2, vlm_tactile_token_history_length=5)
    config.save_pretrained(tmp_path)
    loaded = SmolVLAConfig.from_pretrained(tmp_path)
    assert loaded.get_tactile_history_length() == 2 and loaded.get_tactile_history_length(vlm=True) == 5
    assert replace(legacy, vlm_tactile_token_history_length=3).get_tactile_history_length() == 7
    for name in ("fm_tactile_token_history_length", "vlm_tactile_token_history_length"):
        with pytest.raises(ValueError, match=f"{name} must be positive"):
            replace(legacy, **{name: 0})

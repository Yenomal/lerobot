"""Frame-wise FM-width tactile tokens and history-block inference."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.policies.smolvla.smolvlm_with_expert import FlowTimeRMSNorm
from tests.policies.smolvla import test_delay_adarms

make_model = test_delay_adarms.make_model


@pytest.mark.parametrize("tcn", [False, True])
def test_encoder_width_history_order_bootstrap_and_runtime_reset(make_model, tcn):
    template = make_model(tactile=True, expert_width_multiplier=0.5)
    model = VLAFlowMatching(
        replace(template.config, use_tactile_token_history=tcn, tactile_token_history_length=3)
    )
    policy = object.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.model, policy.config = model, model.config
    policy.config.observation_delay_max_tokens = 0
    policy._tactile_history = {}
    policy.eval()
    assert model.tactile_encoder[0].out_features == 4
    assert model.tactile_encoder[2].in_features == model.tactile_encoder[2].out_features == 4
    assert not hasattr(model, "tactile_token_proj")
    inputs = []
    for step in range(4):
        current = {
            f"observation.tactile.{side}_marker": torch.full((2, 2, 3, 2), float(step))
            for side in ("left", "right")
        }
        inputs.append(current)
        prepared = policy._prepare_batch(dict(current))
        left = prepared["observation.tactile.left_marker_history"]
        assert left[0, :, 0, 0, 0].tolist() == [float(max(0, step - lag)) for lag in (2, 1, 0)]
        actual = model._encode_tactile(prepared)
        # Training gets the same frames stacked on the observation axis.
        training = {
            f"observation.tactile.{side}_marker": prepared[f"observation.tactile.{side}_marker_history"]
            for side in ("left", "right")
        }
        torch.testing.assert_close(actual, model._encode_tactile(training))
        assert actual.shape == ((2, 4) if tcn else (2, 3, 4))
        if not tcn:
            frames = model._encode_tactile_frames(training)
            torch.testing.assert_close(actual, frames)
            if step:
                assert not torch.allclose(actual[:, 0], actual[:, -1])
    policy.reset()
    assert not policy._tactile_history
    prepared = policy._prepare_batch(dict(inputs[-1]))
    assert prepared["observation.tactile.left_marker_history"][0, :, 0, 0, 0].tolist() == [3.0] * 3


@pytest.mark.parametrize("attention_mode", ["self_attn", "cross_attn"])
@pytest.mark.parametrize("tcn,causal", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("noise_adarms", [False, True])
@pytest.mark.parametrize("no_vlm", [False, True])
def test_history_block_noise_delay_layout_cached_parity_and_gradients(
    make_model, attention_mode, tcn, causal, noise_adarms, no_vlm, monkeypatch
):
    torch.manual_seed(56)
    template = make_model(
        attention_mode=attention_mode,
        tactile=True,
        tactile_cross_attention=attention_mode == "cross_attn",
        sincos=True,
        delay_token=True,
        use_noise_adarms=noise_adarms,
    )
    model = VLAFlowMatching(
        replace(
            template.config,
            use_tactile_token_history=tcn,
            tactile_token_history_length=3,
            use_tactile_causal_attention=causal,
            tactile_no_vlm_attention=no_vlm,
        )
    )
    for module in model.modules():
        if isinstance(module, FlowTimeRMSNorm):
            nn.init.normal_(module.projection.weight, std=0.2)
    for layer in model.vlm_with_expert.lm_expert.layers:
        nn.init.normal_(layer.self_attn.o_proj.weight, std=0.5)
    batch = {
        f"observation.tactile.{side}_marker": torch.randn(2, 3, 2, 3, 2, requires_grad=True)
        for side in ("left", "right")
    }
    embedding = model._encode_tactile(batch)
    count = 1 if tcn else 3
    assert model._tactile_token_count(embedding) == count
    actions, noise = torch.randn(2, 4, 6), torch.randn(2, 4, 6)
    time = torch.tensor([[0.2, 0.2, 0.8, 0.8]]).expand(2, -1)
    delay = {"observation_delay_tokens": torch.tensor([0, 2])}
    suffix, _, _ = model.embed_suffix(actions, time, tactile_embedding=embedding, **delay)
    assert suffix.shape == (2, 4 + count + 1 + 4, 8)
    conditions = model._observation_delay_condition(
        2, torch.device("cpu"), action_count=4, tactile_token=count, **delay
    )
    assert conditions.shape == suffix.shape
    if noise_adarms:
        noise_conditions = model._noise_condition(time, 4, tactile_token=count)
        assert noise_conditions.shape == suffix.shape
        torch.testing.assert_close(
            noise_conditions[:, 4 : 4 + count], noise_conditions[:, :1].expand(-1, count, -1)
        )
        torch.testing.assert_close(noise_conditions[:, 4 + count], noise_conditions[:, 5 + count])
    lang, state = torch.tensor([[1, 2], [3, 4]]), torch.randn(2, 6)
    lang_mask = torch.ones_like(lang, dtype=torch.bool)
    cache = model.encode_prefix([], [], lang, lang_mask, state)
    positions = []
    original_forward = model.vlm_with_expert.forward

    def capture_forward(*args, **kwargs):
        positions.append(kwargs["position_ids"].clone())
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.vlm_with_expert, "forward", capture_forward)
    saved = [(layer.keys.clone(), layer.values.clone()) for layer in cache["past_key_values"].layers]
    x_t = time[..., None] * noise + (1 - time[..., None]) * actions
    raw, ca0 = model.denoise_step(
        cache["prefix_pad_masks"],
        cache["past_key_values"],
        x_t,
        time,
        tactile_batch=batch,
        return_tactile_copy=True,
        apply_head_guidance=False,
        **delay,
    )
    loss = model([], [], lang, lang_mask, state, actions, noise, time, tactile_batch=batch, **delay)
    # Cached and joint training paths give CA0/A0 the same position and A_i consecutive positions.
    torch.testing.assert_close(positions[0], positions[1][:, -suffix.shape[1] :])
    first_action = cache["prefix_pad_masks"].sum(-1) + 4 + count
    expected_positions = first_action[:, None] + torch.arange(4)[None]
    torch.testing.assert_close(positions[0][:, -4:], expected_positions)
    torch.testing.assert_close(positions[0][:, -5], positions[0][:, -4])
    expected = (raw - (noise - actions)).square()
    expected[:, 0] = 0.5 * (expected[:, 0] + (ca0 - (noise - actions)[:, 0]).square())
    torch.testing.assert_close(loss, expected)
    for layer, (keys, values) in zip(cache["past_key_values"].layers, saved, strict=True):
        torch.testing.assert_close(layer.keys, keys)
        torch.testing.assert_close(layer.values, values)
    loss.mean().backward()
    assert model.tactile_encoder[0].weight.grad.abs().sum() > 0
    for markers in batch.values():
        assert (markers.grad.abs().flatten(2).sum(-1) > 0).all()


@pytest.mark.parametrize("overrides", [{"use_tactile_token": False}, {"use_tactile_token_history": True}])
def test_causal_config_requires_direct_tactile_tokens(overrides):
    kwargs = {"use_tactile": True, "use_tactile_token": True, "use_tactile_causal_attention": True}
    kwargs.update(overrides)
    with pytest.raises(ValueError, match="use_tactile_causal_attention requires"):
        SmolVLAConfig(**kwargs)


def test_causal_config_round_trip(tmp_path):
    config = SmolVLAConfig(use_tactile=True, use_tactile_token=True, use_tactile_causal_attention=True)
    config.save_pretrained(tmp_path)
    assert SmolVLAConfig.from_pretrained(tmp_path).use_tactile_causal_attention
    assert not SmolVLAConfig().use_tactile_causal_attention


def test_no_vlm_config_validation_and_round_trip(tmp_path):
    with pytest.raises(ValueError, match="tactile_no_vlm_attention requires"):
        SmolVLAConfig(tactile_no_vlm_attention=True)
    config = SmolVLAConfig(use_tactile=True, use_tactile_token=True, tactile_no_vlm_attention=True)
    config.save_pretrained(tmp_path)
    assert SmolVLAConfig.from_pretrained(tmp_path).tactile_no_vlm_attention
    assert not SmolVLAConfig().tactile_no_vlm_attention


@pytest.mark.parametrize("interleave", [0, 2])
@pytest.mark.parametrize("causal", [False, True])
def test_future_tactile_does_not_change_older_causal_hidden_states(
    make_model, monkeypatch, interleave, causal
):
    torch.manual_seed(38)
    model = make_model(
        enabled=False, tactile=True, tactile_cross_attention=True, self_attn_every_n_layers=interleave
    )
    model.config.use_tactile_causal_attention = causal
    for layer in model.vlm_with_expert.lm_expert.layers:
        nn.init.normal_(layer.self_attn.o_proj.weight, std=0.5)
    lang = torch.tensor([[1, 2]])
    cache = model.encode_prefix([], [], lang, torch.ones_like(lang, dtype=torch.bool), torch.randn(1, 6))
    captured = []
    original_forward = model.vlm_with_expert.forward

    def capture_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        captured.append(result[0][1].detach().clone())
        return result

    monkeypatch.setattr(model.vlm_with_expert, "forward", capture_forward)
    embedding = torch.randn(1, 3, 8)
    actions, time = torch.randn(1, 4, 6), torch.full((1, 4), 0.5)

    def predict(tactile):
        return model.denoise_step(
            cache["prefix_pad_masks"],
            cache["past_key_values"],
            actions,
            time,
            tactile_embedding=tactile,
            return_tactile_copy=True,
            apply_head_guidance=False,
        )

    raw, ca0 = predict(embedding)
    changed = embedding.clone()
    changed[:, -1] *= -10
    other_raw, other_ca0 = predict(changed)
    torch.testing.assert_close(raw, other_raw, rtol=0, atol=0)
    assert not torch.allclose(ca0, other_ca0)
    if causal:
        torch.testing.assert_close(captured[0][:, :2], captured[1][:, :2], rtol=0, atol=0)
    else:
        assert not torch.allclose(captured[0][:, :2], captured[1][:, :2])


@pytest.mark.parametrize(
    "attention_mode,interleave,live_keys",
    [
        ("self_attn", 2, False),
        ("cross_attn", 0, False),
        ("cross_attn", 2, False),
        ("cross_attn", 0, True),
        ("cross_attn", 2, True),
    ],
)
@pytest.mark.parametrize("tcn,causal", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("no_vlm", [False, True])
def test_tactile_hidden_independent_of_prefix_when_disabled(
    make_model, monkeypatch, attention_mode, interleave, live_keys, tcn, causal, no_vlm
):
    torch.manual_seed(41)
    template = make_model(
        enabled=False,
        tactile=True,
        attention_mode=attention_mode,
        tactile_cross_attention=live_keys,
        self_attn_every_n_layers=interleave,
    )
    model = VLAFlowMatching(
        replace(
            template.config,
            use_tactile_token_history=tcn,
            tactile_token_history_length=3,
            use_tactile_causal_attention=causal,
            tactile_no_vlm_attention=no_vlm,
        )
    )
    for layer in model.vlm_with_expert.lm_expert.layers:
        nn.init.normal_(layer.self_attn.o_proj.weight, std=0.5)
    lang, state = torch.tensor([[1, 2]]), torch.randn(1, 6)
    valid = torch.ones_like(lang, dtype=torch.bool)
    cache = model.encode_prefix([], [], lang, valid, state)
    changed = model.encode_prefix([], [], torch.tensor([[7, 8]]), valid, -10 * state)
    batch = {f"observation.tactile.{side}_marker": torch.randn(1, 3, 2, 3, 2) for side in ("left", "right")}
    captures = []
    original = model.vlm_with_expert.forward

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captures.append(result[0][1].detach().clone())
        return result

    monkeypatch.setattr(model.vlm_with_expert, "forward", capture)
    actions, noise = torch.randn(1, 4, 6), torch.randn(1, 4, 6)
    time = torch.full((1, 4), 0.5)
    x_t = time[..., None] * noise + (1 - time[..., None]) * actions

    def predict(prefix):
        return model.denoise_step(
            prefix["prefix_pad_masks"],
            prefix["past_key_values"],
            x_t,
            time,
            tactile_batch=batch,
            return_tactile_copy=True,
            apply_head_guidance=False,
        )

    raw, ca0 = predict(cache)
    changed_raw, changed_ca0 = predict(changed)
    assert not torch.allclose(raw, changed_raw)
    assert not torch.allclose(ca0, changed_ca0)
    count = 1 if tcn else 3
    if no_vlm:
        torch.testing.assert_close(captures[0][:, :count], captures[1][:, :count], rtol=0, atol=0)
    else:
        assert not torch.allclose(captures[0][:, :count], captures[1][:, :count])
    losses = model([], [], lang, valid, state, actions, noise, time, tactile_batch=batch)
    expected = (raw - (noise - actions)).square()
    expected[:, 0] = 0.5 * (expected[:, 0] + (ca0 - (noise - actions)[:, 0]).square())
    torch.testing.assert_close(losses, expected)
    torch.testing.assert_close(captures[0][:, :count], captures[2][:, :count])

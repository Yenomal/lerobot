"""Per-layer tactile AdaLN-Zero conditioning, including cached FM inference."""

import pytest
import torch
from torch import nn

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.smolvlm_with_expert import (
    AdaptiveRMSNormZeroModulation,
    FlowTimeRMSNorm,
)
from tests.policies.smolvla import test_delay_adarms

make_model = test_delay_adarms.make_model


def tactile_batch():
    return {f"observation.tactile.{side}_marker": torch.randn(2, 2, 3, 2) for side in ("left", "right")}


@pytest.mark.parametrize("attention_mode", ["self_attn", "cross_attn"])
@pytest.mark.parametrize("other_gate", ["none", "noise", "delay"])
def test_zero_initialization_blocks_residuals_but_gate_gradients_start(
    make_model, attention_mode, other_gate
):
    model = make_model(
        attention_mode,
        enabled=other_gate == "delay",
        adaln_zero=other_gate == "delay",
        use_noise_adarms=other_gate == "noise",
        tactile_adarms=True,
    )
    expert = model.vlm_with_expert
    suffix = torch.randn(2, 4, 8)
    tactile = model._encode_tactile_adarms(tactile_batch())
    positions = torch.arange(6)[None].expand(2, -1)
    prefix = torch.randn(2, 2, 8)
    outputs, _ = expert(
        inputs_embeds=[prefix, suffix],
        position_ids=positions,
        attention_mask=torch.ones(2, 6, 6, dtype=torch.bool),
        tactile_condition=tactile,
        noise_condition=torch.randn(2, 4, 8) if other_gate == "noise" else None,
        expert_condition=torch.randn(2, 8) if other_gate == "delay" else None,
    )
    # At initialization every expert block is the identity, followed by final norm.
    expected = (
        expert.lm_expert.noise_adarms(suffix, torch.zeros_like(suffix))
        if other_gate == "noise"
        else expert.lm_expert.norm(suffix)
    )
    torch.testing.assert_close(outputs[1], expected)
    outputs[1].square().mean().backward()
    for layer in expert.lm_expert.layers:
        for module in layer.tactile_adaln_zero.values():
            assert torch.count_nonzero(module.projection.weight) == 0
            assert torch.count_nonzero(module.projection.bias) == 0
            grad = module.projection.weight.grad
            assert torch.isfinite(grad).all()
            assert grad[16:].abs().sum() > 0
    if other_gate != "none":
        for layer in expert.lm_expert.layers:
            modules = layer.noise_adarms if other_gate == "noise" else layer.observation_delay_adaln_zero
            for module in modules.values():
                assert module.projection.weight.grad[16:].abs().sum() > 0


@pytest.mark.parametrize("attention_mode", ["self_attn", "cross_attn"])
@pytest.mark.parametrize("other_route", ["none", "noise", "delay", "delay_zero"])
@pytest.mark.parametrize("token", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_training_cache_parity_current_tactile_and_gradients(
    make_model, attention_mode, other_route, token, dtype
):
    torch.manual_seed(12)
    model = make_model(
        attention_mode,
        enabled=other_route in ("delay", "delay_zero"),
        adaln_zero=other_route == "delay_zero",
        use_noise_adarms=other_route == "noise",
        tactile_adarms=True,
        tactile=token,
    ).to(dtype)
    for module in model.modules():
        if isinstance(module, (AdaptiveRMSNormZeroModulation, FlowTimeRMSNorm)):
            nn.init.normal_(module.projection.weight, std=0.15)
    # Tiny Llama's default branch outputs can round away in BF16 residuals.
    for layer in model.vlm_with_expert.lm_expert.layers:
        nn.init.normal_(layer.self_attn.o_proj.weight, std=0.5)
        nn.init.normal_(layer.mlp.down_proj.weight, std=0.5)
    lang = torch.tensor([[1, 2], [3, 4]])
    mask = torch.ones_like(lang, dtype=torch.bool)
    state = torch.randn(2, model.config.state_dim).to(dtype)
    actions = torch.randn(2, 4, model.config.action_dim).to(dtype)
    noise = torch.randn_like(actions)
    time = torch.tensor([[0.2, 0.2, 0.8, 0.8]]).expand(2, -1)
    x_t = (time[..., None] * noise + (1 - time[..., None]) * actions).to(dtype)
    batch = tactile_batch()
    delays = {"observation_delay_tokens": torch.tensor([0, 3])}
    cache = model.encode_prefix([], [], lang, mask, state)
    keys = [layer.keys.clone() for layer in cache["past_key_values"].layers]
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        velocity = model.denoise_step(
            cache["prefix_pad_masks"],
            cache["past_key_values"],
            x_t,
            time,
            tactile_batch=batch,
            apply_head_guidance=False,
            return_tactile_copy=token,
            **delays,
        )
        if token:
            velocity, copy = velocity
        losses = model(
            [], [], lang, mask, state, actions, noise=noise, time=time, tactile_batch=batch, **delays
        )
        expected = nn.functional.mse_loss(velocity, noise - actions, reduction="none")
        if token:
            expected[:, 0] = 0.5 * (
                expected[:, 0] + nn.functional.mse_loss(copy, noise[:, 0] - actions[:, 0], reduction="none")
            )
        torch.testing.assert_close(losses, expected)
        changed = model.denoise_step(
            cache["prefix_pad_masks"],
            cache["past_key_values"],
            x_t,
            time,
            tactile_batch={key: -10 * value for key, value in batch.items()},
            apply_head_guidance=False,
            **delays,
        )
    assert not torch.allclose(velocity, changed)
    for key, layer in zip(keys, cache["past_key_values"].layers, strict=True):
        torch.testing.assert_close(key, layer.keys)
    losses.mean().backward()
    assert model.tactile_encoder[0].weight.grad.abs().sum() > 0
    assert model.tactile_adarms[0].weight.grad.abs().sum() > 0
    for layer in model.vlm_with_expert.lm_expert.layers:
        for module in layer.tactile_adaln_zero.values():
            assert module.projection.weight.grad.abs().sum() > 0


def test_peft_trains_zero_projections_in_full(make_model):
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.model = make_model(enabled=False, tactile_adarms=True)
    policy.config = policy.model.config
    saved = policy._get_default_peft_targets()["modules_to_save"]
    assert len(saved) == 4
    assert all(".tactile_adaln_zero." in name for name in saved)

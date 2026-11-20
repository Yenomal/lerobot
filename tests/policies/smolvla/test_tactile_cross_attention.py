"""Live FM tactile keys in cross-attention never enter the reusable VLM cache."""

import pytest
import torch
from torch import nn

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from tests.policies.smolvla import test_delay_adarms

make_model = test_delay_adarms.make_model


@pytest.mark.parametrize("count_bias", [False, True])
@pytest.mark.parametrize("width", [0.5, 1.0])
@pytest.mark.parametrize("delay", [False, True])
@pytest.mark.parametrize("interleave", [0, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("history_count", [1, 3])
def test_live_tactile_visibility_training_cache_parity_and_gradients(
    make_model, width, delay, interleave, dtype, count_bias, history_count
):
    torch.manual_seed(28)
    model = make_model(
        enabled=False,
        tactile=True,
        tactile_cross_attention=True,
        modality_count_bias=count_bias,
        delay_token=delay,
        sincos=delay,
        expert_width_multiplier=width,
        self_attn_every_n_layers=interleave,
    ).to(dtype)
    model.config.tactile_token_history_length = history_count
    # Give the tiny initialized model observable residuals even in BF16.
    for layer in model.vlm_with_expert.lm_expert.layers:
        nn.init.normal_(layer.self_attn.o_proj.weight, std=0.5)
    batch = {
        f"observation.tactile.{side}_marker": torch.randn(2, history_count, 2, 3, 2)
        for side in ("left", "right")
    }
    lang = torch.tensor([[1, 2], [3, 4]])
    mask = torch.ones_like(lang, dtype=torch.bool)
    state = torch.randn(2, model.config.state_dim).to(dtype)
    actions = torch.randn(2, 4, model.config.action_dim).to(dtype)
    noise = torch.randn_like(actions)
    time = torch.tensor([[0.2, 0.2, 0.8, 0.8]]).expand(2, -1)
    x_t = (time[..., None] * noise + (1 - time[..., None]) * actions).to(dtype)
    cache = model.encode_prefix([], [], lang, mask, state)
    cached_keys = [layer.keys.clone() for layer in cache["past_key_values"].layers]
    cached_values = [layer.values.clone() for layer in cache["past_key_values"].layers]
    captured = []
    expert = model.vlm_with_expert
    original_attention = expert.get_attention_interface()

    def attention(attention_mask, batch_size, head_dim, query, key, value, bias=None):
        captured.append((attention_mask.detach().clone(), key.detach().clone()))
        return original_attention(attention_mask, batch_size, head_dim, query, key, value, bias)

    expert.get_attention_interface = lambda: attention
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        velocity, copy = model.denoise_step(
            cache["prefix_pad_masks"],
            cache["past_key_values"],
            x_t,
            time,
            tactile_batch=batch,
            apply_head_guidance=False,
            return_tactile_copy=True,
        )
        # Cross layers use prefix + the live tactile block. Only T and CA0
        # see the block; real actions and delay queries remain isolated.
        prefix_len = cache["prefix_pad_masks"].shape[1]
        cross_calls = [(m, k) for m, k in captured if k.shape[1] == prefix_len + history_count]
        assert len(cross_calls) == (2 if interleave == 0 else 1)
        tactile_index = model._delay_token_count(4)
        for m, _ in cross_calls:
            assert not m[:, :tactile_index, -history_count:].any()
            assert m[:, tactile_index : tactile_index + history_count + 1, -history_count:].all()
            assert not m[:, tactile_index + history_count + 1 :, -history_count:].any()
        loss = model([], [], lang, mask, state, actions, noise=noise, time=time, tactile_batch=batch)
        expected = nn.functional.mse_loss(velocity, noise - actions, reduction="none")
        expected[:, 0] = 0.5 * (
            expected[:, 0] + nn.functional.mse_loss(copy, noise[:, 0] - actions[:, 0], reduction="none")
        )
        torch.testing.assert_close(loss, expected)
        changed, changed_copy = model.denoise_step(
            cache["prefix_pad_masks"],
            cache["past_key_values"],
            x_t,
            time,
            tactile_batch={key: -10 * value for key, value in batch.items()},
            apply_head_guidance=False,
            return_tactile_copy=True,
        )
    # Fresh tactile changes only CA0 across the entire stack, including
    # mixed self/cross layers. Real actions never receive it indirectly.
    torch.testing.assert_close(velocity, changed, rtol=0, atol=0)
    assert not torch.allclose(copy, changed_copy)
    for keys, values, layer in zip(cached_keys, cached_values, cache["past_key_values"].layers, strict=True):
        torch.testing.assert_close(keys, layer.keys)
        torch.testing.assert_close(values, layer.values)
    loss.mean().backward()
    for layer in expert.lm_expert.layers:
        if hasattr(layer, "tactile_cross_k"):
            assert layer.tactile_cross_k.weight.grad.abs().sum() > 0
            assert layer.tactile_cross_v.weight.grad.abs().sum() > 0
    assert model.tactile_encoder[0].weight.grad.abs().sum() > 0


def test_cross_tactile_does_not_change_vlm_outputs(make_model):
    model = make_model(enabled=False, tactile=True, tactile_cross_attention=True)
    expert = model.vlm_with_expert
    prefix = torch.randn(2, 3, 8)
    suffix = torch.randn(2, 6, 8)
    # Prefix cannot see suffix; every suffix query sees the prefix.
    mask = torch.ones(2, 9, 9, dtype=torch.bool)
    mask[:, :3, 3:] = False
    positions = torch.arange(9)[None].expand(2, -1)
    first, _ = expert(
        inputs_embeds=[prefix, suffix], attention_mask=mask, position_ids=positions, tactile_token_index=0
    )
    changed = suffix.clone()
    changed[:, 0] *= -10
    second, _ = expert(
        inputs_embeds=[prefix, changed], attention_mask=mask, position_ids=positions, tactile_token_index=0
    )
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    assert not torch.allclose(first[1], second[1])


def test_disabled_adds_no_parameters_and_peft_saves_new_projections(make_model):
    disabled = make_model(enabled=False, tactile=True)
    assert not any("tactile_cross_" in name for name, _ in disabled.named_parameters())
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.model = make_model(enabled=False, tactile=True, tactile_cross_attention=True)
    policy.config = policy.model.config
    saved = policy._get_default_peft_targets()["modules_to_save"]
    assert len(saved) == 2
    assert all(".tactile_cross_" in name for name in saved)


def test_config_validation_and_round_trip(tmp_path):
    with pytest.raises(ValueError, match="use_tactile_cross_attention requires"):
        SmolVLAConfig(use_tactile_cross_attention=True)
    with pytest.raises(ValueError, match="use_tactile_cross_attention requires"):
        SmolVLAConfig(
            use_tactile=True,
            use_tactile_token=True,
            use_tactile_cross_attention=True,
            attention_mode="self_attn",
        )
    config = SmolVLAConfig(
        device="cpu", use_tactile=True, use_tactile_token=True, use_tactile_cross_attention=True
    )
    config.save_pretrained(tmp_path)
    assert SmolVLAConfig.from_pretrained(tmp_path).use_tactile_cross_attention

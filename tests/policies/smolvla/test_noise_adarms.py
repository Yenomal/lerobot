# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest
import torch
from torch import nn

from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import FlowTimeRMSNorm
from tests.policies.smolvla import test_delay_adarms

make_model = test_delay_adarms.make_model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("per_token", [False, True])
def test_flow_norm_matches_reference_equations(dtype, per_token):
    torch.manual_seed(3)
    norm = FlowTimeRMSNorm(8).to(dtype)
    x = torch.randn(2, 4, 8).to(dtype)
    condition = torch.randn((2, 4, 8) if per_token else (2, 8)).to(dtype)
    rms = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(norm(x, condition), rms.to(dtype))
    torch.testing.assert_close(norm.gate(x, condition), torch.zeros_like(x))
    nn.init.normal_(norm.projection.weight, std=0.2)
    nn.init.normal_(norm.projection.bias, std=0.1)
    modulation = nn.functional.linear(condition, norm.projection.weight, norm.projection.bias)
    if not per_token:
        modulation = modulation[:, None, :]
    scale, shift, gate = modulation.chunk(3, -1)
    expected = (rms * (1 + scale.float()) + shift.float()).to(dtype)
    torch.testing.assert_close(norm(x, condition), expected)
    torch.testing.assert_close(norm.gate(x, condition), x * gate)


@pytest.mark.parametrize("delay_token", [False, True])
@pytest.mark.parametrize("tactile", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
def test_time_condition_matches_reference_mlp_and_suffix_layout(make_model, delay_token, tactile, per_token):
    model = make_model(
        enabled=False,
        use_noise_embedding=False,
        use_noise_adarms=True,
        delay_token=delay_token,
        sincos=delay_token,
        tactile=tactile,
    )
    time = torch.tensor([[0.2, 0.2, 0.8, 0.8]]).expand(2, -1) if per_token else torch.tensor([0.2, 0.8])
    x = torch.randn(2, 4, model.config.action_dim)
    tactile_embedding = torch.randn(2, 8) if tactile else None
    suffix, _, _ = model.embed_suffix(x, time, tactile_embedding=tactile_embedding)
    # Reference action tokens are just the projected noisy actions.
    torch.testing.assert_close(suffix[:, -4:], model.action_in_proj(x))
    condition = model._noise_condition(time, 4, tactile_token=tactile)
    assert condition.shape == suffix.shape

    def reference_mlp(t):
        encoding = create_sinusoidal_pos_embedding(t, 8, 0.004, 4.0, t.device).float()
        return nn.functional.silu(model.time_mlp_out(nn.functional.silu(model.time_mlp_in(encoding))))

    expected = reference_mlp(time)
    if not per_token:
        expected = expected[:, None, :].expand(-1, 4, -1)
    torch.testing.assert_close(condition[:, -4:], expected)
    if tactile:
        torch.testing.assert_close(condition[:, -5], condition[:, -4])  # CA0 = A0.
    static_count = (4 if delay_token else 0) + int(tactile)
    if static_count:
        clean = reference_mlp(torch.zeros(2))[:, None, :].expand(-1, static_count, -1)
        torch.testing.assert_close(condition[:, :static_count], clean)
    if per_token:
        torch.testing.assert_close(condition[:, -4], condition[:, -3])
        assert not torch.allclose(condition[:, -3], condition[:, -2])
        changed = time.clone()
        changed[:, 2:] = 0.6
        changed_condition = model._noise_condition(changed, 4, tactile_token=tactile)
        torch.testing.assert_close(condition[:, :-2], changed_condition[:, :-2])
        assert not torch.allclose(condition[:, -2:], changed_condition[:, -2:])


@pytest.mark.parametrize("attention_mode", ["self_attn", "cross_attn"])
@pytest.mark.parametrize("delay_route", ["none", "adarms", "zero", "token"])
@pytest.mark.parametrize("tactile", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_noise_adarms_training_cache_parity_and_gradients(
    make_model, attention_mode, delay_route, tactile, dtype
):
    torch.manual_seed(4)
    model = make_model(
        attention_mode,
        enabled=delay_route in ("adarms", "zero"),
        adaln_zero=delay_route == "zero",
        delay_token=delay_route == "token",
        sincos=delay_route != "none",
        tactile=tactile,
        use_noise_embedding=False,
        use_noise_adarms=True,
    ).to(dtype)
    # Learned nonzero projections exercise attention/MLP, gates and final norm.
    for module in model.modules():
        if isinstance(module, FlowTimeRMSNorm):
            nn.init.normal_(module.projection.weight, std=0.1)
    lang = torch.tensor([[1, 2], [3, 4]])
    mask = torch.ones_like(lang, dtype=torch.bool)
    state = torch.randn(2, model.config.state_dim).to(dtype)
    actions = torch.randn(2, 4, model.config.action_dim).to(dtype)
    noise = torch.randn_like(actions)
    time = torch.tensor([[0.2, 0.2, 0.8, 0.8]]).expand(2, -1)
    x_t = (time[..., None] * noise + (1 - time[..., None]) * actions).to(dtype)
    tactile_batch = {} if tactile else None
    if tactile:
        embedding = torch.randn(2, 8).to(dtype)
        model._encode_tactile = lambda batch: embedding
    delays = {"observation_delay_tokens": torch.tensor([0, 4])}
    cache = model.encode_prefix([], [], lang, mask, state)
    keys = [layer.keys.clone() for layer in cache["past_key_values"].layers]
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        velocity = model.denoise_step(
            cache["prefix_pad_masks"],
            cache["past_key_values"],
            x_t,
            time,
            tactile_batch=tactile_batch,
            apply_head_guidance=False,
            return_tactile_copy=tactile,
            **delays,
        )
        if tactile:
            velocity, copy = velocity
        loss = model(
            [],
            [],
            lang,
            mask,
            state,
            actions,
            noise=noise,
            time=time,
            tactile_batch=tactile_batch,
            **delays,
        )
        expected = nn.functional.mse_loss(velocity, noise - actions, reduction="none")
        if tactile:
            expected[:, 0] = 0.5 * (
                expected[:, 0] + nn.functional.mse_loss(copy, noise[:, 0] - actions[:, 0], reduction="none")
            )
        torch.testing.assert_close(loss, expected)
        changed = model.denoise_step(
            cache["prefix_pad_masks"],
            cache["past_key_values"],
            x_t,
            time * 0.5,
            tactile_batch=tactile_batch,
            apply_head_guidance=False,
            **delays,
        )
        assert not torch.allclose(velocity, changed)
    for expected_keys, layer in zip(keys, cache["past_key_values"].layers, strict=True):
        torch.testing.assert_close(layer.keys, expected_keys)
    loss.mean().backward()
    assert model.time_mlp_in.weight.grad.abs().sum() > 0
    assert model.time_mlp_out.weight.grad.abs().sum() > 0
    for module in model.modules():
        if isinstance(module, FlowTimeRMSNorm):
            assert module.projection.weight.grad.abs().sum() > 0
    assert model.action_time_mlp_in.weight.grad is None
    assert not model.action_time_mlp_in.weight.requires_grad
    if model.observation_delay_adarms is not None:
        assert model.observation_delay_adarms[0].weight.grad.abs().sum() > 0


def test_noise_adarms_config_round_trip_and_legacy_default(tmp_path):
    config = SmolVLAConfig(device="cpu", use_noise_adarms=True, use_noise_embedding=False)
    config.save_pretrained(tmp_path)
    restored = SmolVLAConfig.from_pretrained(tmp_path)
    assert restored.use_noise_adarms and not restored.use_noise_embedding
    path = tmp_path / "config.json"
    legacy = json.loads(path.read_text())
    del legacy["use_noise_adarms"]
    path.write_text(json.dumps(legacy))
    assert not SmolVLAConfig.from_pretrained(tmp_path).use_noise_adarms


@pytest.mark.parametrize("mode", ["diffusion", "stream", "stream-head", "stream-head-mean-tactile-cfg"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sampling_refreshes_chunk_times_in_every_norm(make_model, mode, dtype):
    tactile = "tactile" in mode
    model = (
        make_model(
            enabled=False,
            use_noise_embedding=False,
            use_noise_adarms=True,
            delay_token=True,
            sincos=True,
            tactile=tactile,
        )
        .to(dtype)
        .eval()
    )
    model.reset_stream_head_state()
    if tactile:
        model._encode_tactile = lambda batch: torch.ones(1, 8).to(dtype)
    cache = model.encode_prefix(
        [],
        [],
        torch.tensor([[1, 2]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.randn(1, model.config.state_dim).to(dtype),
    )
    observed_times = []
    conditions = []
    encode_condition = model._noise_condition

    def record(time, count, **kwargs):
        condition = encode_condition(time, count, **kwargs)
        observed_times.append(time.clone())
        conditions.append(condition)
        return condition

    model._noise_condition = record
    norms_called = set()

    def check_condition(module, args):
        torch.testing.assert_close(args[1], conditions[-1])
        norms_called.add(id(module))

    norms = [module for module in model.modules() if isinstance(module, FlowTimeRMSNorm)]
    handles = [norm.register_forward_pre_hook(check_condition) for norm in norms]
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            for _ in range(2):
                result = model.sample_actions_from_prefix(
                    cache,
                    inference_mode=mode,
                    tactile_batch={} if tactile else None,
                )
                assert result.shape == (1, 4, model.config.action_dim)
                assert torch.isfinite(result).all()
        assert len(norms_called) == len(norms)
        if mode == "diffusion":
            assert all(time.ndim == 1 for time in observed_times)
        else:
            assert observed_times[-1].shape == (1, 4)
            assert not torch.equal(conditions[-1][:, -4:-2], conditions[-1][:, -2:])
    finally:
        for handle in handles:
            handle.remove()


def test_zero_initialized_gates_receive_gradients_and_unused_static_norms_are_frozen(make_model):
    model = make_model(enabled=False, use_noise_embedding=False, use_noise_adarms=True)
    actions = torch.randn(2, 4, model.config.action_dim)
    loss = model(
        [],
        [],
        torch.tensor([[1, 2], [3, 4]]),
        torch.ones(2, 2, dtype=torch.bool),
        torch.randn(2, model.config.state_dim),
        actions,
    )
    loss.mean().backward()
    for layer in model.vlm_with_expert.lm_expert.layers:
        assert not layer.input_layernorm.weight.requires_grad
        assert not layer.post_attention_layernorm.weight.requires_grad
        for norm in layer.noise_adarms.values():
            assert norm.projection.weight.grad[16:].abs().sum() > 0
    assert all(param.grad is not None for param in model.parameters() if param.requires_grad)

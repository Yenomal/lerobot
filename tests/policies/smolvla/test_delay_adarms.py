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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla import smolvlm_with_expert
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.policies.smolvla.smolvlm_with_expert import (
    AdaptiveRMSNormModulation,
    AdaptiveRMSNormZeroModulation,
)

transformers = pytest.importorskip("transformers")


@pytest.fixture
def make_model(monkeypatch):
    """Exercise the real FM constructor and layers without downloading a VLM."""
    text_config = transformers.LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=2,
        vocab_size=16,
    )
    config = SimpleNamespace(text_config=text_config)

    def tiny_vlm(config):
        vlm = nn.Module()
        vlm.config = config
        vlm.model = nn.Module()
        vlm.model.text_model = transformers.LlamaModel(config.text_config)
        return vlm

    monkeypatch.setattr(smolvlm_with_expert.AutoConfig, "from_pretrained", lambda _: config)
    monkeypatch.setattr(smolvlm_with_expert, "SmolVLMForConditionalGeneration", tiny_vlm)
    monkeypatch.setattr(
        smolvlm_with_expert.AutoProcessor,
        "from_pretrained",
        lambda _: SimpleNamespace(tokenizer=SimpleNamespace(fake_image_token_id=0, global_image_token_id=1)),
    )

    def make(
        attention_mode="cross_attn",
        enabled=True,
        delay_unit="tokens",
        use_noise_embedding=True,
        adaln_zero=False,
        sincos=False,
        embedding=False,
        delay_token=False,
        tactile=False,
        use_noise_adarms=False,
        zero_bypass=False,
    ):
        return VLAFlowMatching(
            SmolVLAConfig(
                device="cpu",
                streaming=True,
                chunk_size=4,
                n_action_steps=4,
                streaming_chunk_size=2,
                observation_delay_max_tokens=4 if delay_unit == "tokens" else 0,
                observation_delay_max_chunks=4 if delay_unit == "chunks" else 0,
                use_observation_delay_adarms=enabled and not adaln_zero,
                use_observation_delay_adaln_zero=enabled and adaln_zero,
                observation_delay_zero_bypass=zero_bypass,
                use_observation_delay_sincos=sincos,
                use_observation_delay_embedding=embedding,
                use_observation_delay_token=delay_token,
                use_tactile=tactile,
                use_tactile_token=tactile,
                use_noise_embedding=use_noise_embedding,
                use_noise_adarms=use_noise_adarms,
                freeze_vision_encoder=False,
                expert_width_multiplier=1.0,
                attention_mode=attention_mode,
                num_vlm_layers=2,
                num_expert_layers=2,
                self_attn_every_n_layers=2,
            )
        )

    return make


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_modulation_starts_as_identity_and_applies_scale_and_shift(dtype):
    modulation = AdaptiveRMSNormModulation(8).to(dtype)
    normalized = nn.RMSNorm(8)(torch.randn(2, 4, 8)).to(dtype)
    condition = torch.randn(2, 8)
    torch.testing.assert_close(modulation(normalized, condition), normalized)
    with torch.no_grad():
        modulation.projection.bias[:8].fill_(0.5)
        modulation.projection.bias[8:].fill_(0.25)
    torch.testing.assert_close(modulation(normalized, condition), normalized * 1.5 + 0.25)


@pytest.mark.parametrize("attention_mode", ["self_attn", "cross_attn"])
@pytest.mark.parametrize("delay_unit", ["tokens", "chunks"])
@pytest.mark.parametrize("use_noise_embedding", [True, False])
@pytest.mark.parametrize("adaln_zero", [False, True])
def test_delay_modulates_expert_with_training_cache_parity_and_gradients(
    make_model, attention_mode, delay_unit, use_noise_embedding, adaln_zero
):
    torch.manual_seed(7)
    model = make_model(
        attention_mode, delay_unit=delay_unit, use_noise_embedding=use_noise_embedding, adaln_zero=adaln_zero
    )
    # Simulate learned modulation so different delays have an observable effect.
    for module in model.modules():
        if isinstance(module, (AdaptiveRMSNormModulation, AdaptiveRMSNormZeroModulation)):
            nn.init.normal_(module.projection.weight, std=0.1)
    lang_tokens = torch.tensor([[1, 2], [3, 4]])
    lang_masks = torch.ones_like(lang_tokens, dtype=torch.bool)
    state = torch.randn(2, model.config.state_dim)
    actions = torch.randn(2, 4, model.config.action_dim)
    noise = torch.randn_like(actions)
    time = torch.tensor([[0.2, 0.2, 0.8, 0.8]]).expand(2, -1)
    x_t = time[..., None] * noise + (1 - time[..., None]) * actions
    delay_key = f"observation_delay_{delay_unit}"
    delays = {delay_key: torch.tensor([0, 4])}
    no_delay = {delay_key: torch.zeros(2, dtype=torch.long)}

    suffix, _, _ = model.embed_suffix(x_t, time, **delays)
    suffix_zero, _, _ = model.embed_suffix(x_t, time, **no_delay)
    torch.testing.assert_close(suffix, suffix_zero)  # No input-embedding addition.

    # A prefix prefilled with delay=0 must work with a different current delay.
    cache = model.encode_prefix([], [], lang_tokens, lang_masks, state, **no_delay)
    cached_keys = [layer.keys.clone() for layer in cache["past_key_values"].layers]
    velocity = model.denoise_step(
        cache["prefix_pad_masks"],
        cache["past_key_values"],
        x_t,
        time,
        apply_head_guidance=False,
        **delays,
    )
    loss = model([], [], lang_tokens, lang_masks, state, actions, noise=noise, time=time, **delays)
    torch.testing.assert_close(loss, (noise - actions - velocity).square())
    velocity_zero = model.denoise_step(
        cache["prefix_pad_masks"],
        cache["past_key_values"],
        x_t,
        time,
        apply_head_guidance=False,
        **no_delay,
    )
    torch.testing.assert_close(velocity[0], velocity_zero[0])
    assert not torch.allclose(velocity[1], velocity_zero[1])
    for expected, layer in zip(cached_keys, cache["past_key_values"].layers, strict=True):
        torch.testing.assert_close(layer.keys, expected)

    loss.mean().backward()
    for layer in model.vlm_with_expert.lm_expert.layers:
        modulations = layer.observation_delay_adaln_zero if adaln_zero else layer.observation_delay_adarms
        for modulation in modulations.values():
            assert modulation.projection.weight.grad.abs().sum() > 0
        if adaln_zero:
            assert layer.input_layernorm.weight.grad.abs().sum() > 0
            assert layer.post_attention_layernorm.weight.grad.abs().sum() > 0
    assert model.observation_delay_adarms[0].weight.grad.abs().sum() > 0


def test_disabled_delay_adds_no_modulation_parameters(make_model):
    model = make_model(enabled=False)
    assert model.observation_delay_adarms is None
    assert not any(isinstance(module, AdaptiveRMSNormModulation) for module in model.modules())
    assert model._observation_delay_condition(2, torch.device("cpu")) is None


@pytest.mark.parametrize("unit", ["tokens", "chunks"])
@pytest.mark.parametrize("mode", ["stream-head", "stream-head-context-cfg", "stream-head-mean"])
@pytest.mark.parametrize("adaln_zero", [False, True])
@pytest.mark.parametrize("sincos", [False, True])
def test_stream_head_uses_current_batch_delay_in_all_expert_modulations(
    make_model, unit, mode, adaln_zero, sincos
):
    model = make_model(
        delay_unit=unit, adaln_zero=adaln_zero, sincos=sincos, delay_token=sincos, tactile=sincos
    ).eval()
    if sincos:
        model._encode_tactile = lambda batch: torch.ones(1, model.vlm_with_expert.expert_hidden_size)
    model.config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(model.config.action_dim,))
    }
    # Use the real policy prediction path around the tiny, real FM/expert model.
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = model.config
    policy.model = model
    policy.eval()
    key = f"observation_delay_{unit}"
    state = torch.randn(1, model.config.state_dim)
    cache = model.encode_prefix(
        [],
        [],
        torch.tensor([[1, 2]]),
        torch.ones(1, 2, dtype=torch.bool),
        state,
        **{key: torch.zeros(1, dtype=torch.long)},
    )
    cached_keys = [layer.keys.clone() for layer in cache["past_key_values"].layers]
    encoded_delays = []
    conditions = []
    handles = [
        model.observation_delay_adarms[0].register_forward_pre_hook(
            lambda module, args: encoded_delays.append(args[0].detach().clone())
        )
    ]
    for module in model.modules():
        if isinstance(module, (AdaptiveRMSNormModulation, AdaptiveRMSNormZeroModulation)):
            nn.init.normal_(module.projection.weight, std=0.1)
            handles.append(
                module.register_forward_pre_hook(
                    lambda module, args: conditions.append(args[1].detach().clone())
                )
            )
    try:
        with torch.no_grad():
            for index, delay in enumerate((2, 4, 1)):
                encoded_delays.clear()
                conditions.clear()
                actions = policy.predict_action_chunk_from_prefix(
                    {"observation.state": state, key: torch.tensor([delay])},
                    cache,
                    inference_mode=mode,
                )
                # First invocation includes full-diffusion warmup; later calls reuse the window.
                calls = model.config.num_steps + 1 if index == 0 else 1
                assert len(encoded_delays) == calls
                features = (
                    model._delay_position_encoding(torch.tensor([delay]), model.config.chunk_size)
                    if sincos
                    else torch.tensor([[delay / 4]])
                )
                for value in encoded_delays:
                    torch.testing.assert_close(value, features)
                encoded = model.observation_delay_adarms(features)
                expected = (
                    torch.cat([encoded, encoded[:, :1].expand(-1, 2, -1), encoded], dim=1)
                    if sincos
                    else encoded
                )
                assert len(conditions) == calls * model.vlm_with_expert.num_expert_layers * 2
                for condition in conditions:
                    torch.testing.assert_close(condition, expected)
                assert torch.isfinite(actions).all()
                for original, layer in zip(cached_keys, cache["past_key_values"].layers, strict=True):
                    torch.testing.assert_close(layer.keys, original)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adaln_zero_modulates_normalized_features_and_gates_each_residual(dtype):
    modulation = AdaptiveRMSNormZeroModulation(8).to(dtype)
    hidden = torch.randn(2, 4, 8).to(dtype)
    condition = torch.randn(2, 8).to(dtype)
    normalized = nn.RMSNorm(8, eps=1e-6).to(dtype)(hidden)
    torch.testing.assert_close(modulation(normalized, condition), normalized)
    torch.testing.assert_close(modulation.gate(hidden, condition), torch.zeros_like(hidden))
    with torch.no_grad():
        modulation.projection.bias[:8].fill_(0.5)
        modulation.projection.bias[8:16].fill_(0.25)
        modulation.projection.bias[16:].fill_(0.75)
    torch.testing.assert_close(modulation(normalized, condition), normalized * 1.5 + 0.25)
    torch.testing.assert_close(modulation.gate(hidden, condition), hidden * 0.75)


@pytest.mark.parametrize("branch", ["attention", "mlp"])
def test_adaln_zero_keeps_original_rmsnorm_weights_and_epsilon(make_model, branch):
    model = make_model(adaln_zero=True)
    layer = model.vlm_with_expert.lm_expert.layers[0]
    norm = layer.input_layernorm if branch == "attention" else layer.post_attention_layernorm
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, norm.weight.numel()))
    epsilon = norm.variance_epsilon
    hidden = torch.randn(2, 4, model.vlm_with_expert.expert_hidden_size)
    condition = torch.randn(2, model.vlm_with_expert.expert_hidden_size)
    actual = model.vlm_with_expert._conditioned_norm(layer, hidden, condition, branch)
    torch.testing.assert_close(actual, norm(hidden))
    assert norm.variance_epsilon == epsilon
    assert norm.weight.requires_grad


@pytest.mark.parametrize("attention_mode", ["self_attn", "cross_attn"])
def test_adaln_zero_starts_with_identity_expert_blocks_and_learns_gates(make_model, attention_mode):
    model = make_model(attention_mode=attention_mode, adaln_zero=True)
    state = torch.randn(1, model.config.state_dim)
    cache = model.encode_prefix([], [], torch.tensor([[1, 2]]), torch.ones(1, 2, dtype=torch.bool), state)
    x_t = torch.randn(1, model.config.chunk_size, model.config.action_dim)
    time = torch.tensor([0.5])
    suffix, _, _ = model.embed_suffix(x_t, time)
    expected = model.action_out_proj(model.vlm_with_expert.lm_expert.norm(suffix))
    actual = model.denoise_step(
        cache["prefix_pad_masks"],
        cache["past_key_values"],
        x_t,
        time,
        observation_delay_tokens=torch.tensor([3]),
        apply_head_guidance=False,
    )
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    width = model.vlm_with_expert.expert_hidden_size
    for layer in model.vlm_with_expert.lm_expert.layers:
        assert layer.input_layernorm.weight.requires_grad
        assert layer.post_attention_layernorm.weight.requires_grad
        for modulation in layer.observation_delay_adaln_zero.values():
            grad = modulation.projection.weight.grad
            assert grad[2 * width :].abs().sum() > 0  # Gates learn on the first update.
            assert grad[: 2 * width].abs().sum() == 0  # Scale/shift wait for gates to open.
    assert model.observation_delay_adarms[0].weight.grad.abs().sum() == 0


def test_adaln_zero_config_requires_delay_and_excludes_adarms():
    with pytest.raises(ValueError, match="non-zero observation delay limit"):
        SmolVLAConfig(use_observation_delay_adaln_zero=True)
    with pytest.raises(ValueError, match="cannot both be enabled"):
        SmolVLAConfig(
            streaming=True,
            observation_delay_max_tokens=4,
            use_observation_delay_adarms=True,
            use_observation_delay_adaln_zero=True,
        )


@pytest.mark.parametrize("unit", ["tokens", "chunks"])
@pytest.mark.parametrize("sincos", [False, True])
@pytest.mark.parametrize("attention_mode", ["self_attn", "cross_attn"])
def test_zero_bypass_matches_original_expert_for_zero_age_and_preserves_delayed_samples(
    make_model, unit, sincos, attention_mode
):
    model = make_model(
        attention_mode=attention_mode,
        delay_unit=unit,
        sincos=sincos,
        adaln_zero=True,
        zero_bypass=True,
    ).eval()
    for module in model.modules():
        if isinstance(module, AdaptiveRMSNormZeroModulation):
            nn.init.normal_(module.projection.weight, std=0.1)
            nn.init.normal_(module.projection.bias, std=0.1)
    state = torch.randn(2, model.config.state_dim)
    lang = torch.tensor([[1, 2], [3, 4]])
    masks = torch.ones_like(lang, dtype=torch.bool)
    cache = model.encode_prefix([], [], lang, masks, state)
    actions = torch.randn(2, 4, model.config.action_dim)
    noise = torch.randn_like(actions)
    time = torch.tensor([0.5, 0.5])
    x_t = 0.5 * actions + 0.5 * noise
    delay_kw = {f"observation_delay_{unit}": torch.tensor([0, 3])}

    def velocity():
        return model.denoise_step(
            cache["prefix_pad_masks"],
            cache["past_key_values"],
            x_t,
            time,
            apply_head_guidance=False,
            **delay_kw,
        )

    encoder = model.observation_delay_adarms
    actual = velocity()
    model.observation_delay_adarms = None
    original = velocity()
    model.observation_delay_adarms = encoder
    torch.testing.assert_close(actual[0], original[0], rtol=0, atol=0)
    assert not torch.allclose(actual[1], original[1])
    model.config.observation_delay_zero_bypass = False
    legacy = velocity()
    torch.testing.assert_close(actual[1], legacy[1], rtol=0, atol=0)
    assert not torch.allclose(actual[0], legacy[0])
    model.config.observation_delay_zero_bypass = True
    loss = model([], [], lang, masks, state, actions, noise=noise, time=time, **delay_kw)
    torch.testing.assert_close(loss, (noise - actions - actual).square())
    loss[0].mean().backward()
    for module in model.modules():
        if isinstance(module, AdaptiveRMSNormZeroModulation):
            assert module.projection.weight.grad.abs().sum() == 0
            assert module.projection.bias.grad.abs().sum() == 0
    assert model.observation_delay_adarms[0].weight.grad.abs().sum() == 0
    assert model.vlm_with_expert.lm_expert.layers[0].input_layernorm.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["diffusion", "stream-head"])
def test_zero_bypass_updates_from_current_delay_in_cached_sampling(make_model, monkeypatch, mode):
    model = make_model(adaln_zero=True, zero_bypass=True, sincos=True).eval()
    model.reset_stream_head_state()
    cache = model.encode_prefix(
        [],
        [],
        torch.tensor([[1, 2]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.randn(1, model.config.state_dim),
        observation_delay_tokens=torch.tensor([0]),
    )
    seen = []
    original = model._observation_delay_modulation_mask

    def capture(*args, **kwargs):
        mask = original(*args, **kwargs)
        seen.append(mask.clone())
        return mask

    monkeypatch.setattr(model, "_observation_delay_modulation_mask", capture)
    with torch.no_grad():
        for delay in (3, 0):
            seen.clear()
            model.sample_actions_from_prefix(
                cache,
                inference_mode=mode,
                observation_delay_tokens=torch.tensor([delay]),
            )
            assert seen
            assert all(mask.tolist() == [delay > 0] for mask in seen)


def test_zero_bypass_requires_independent_delay_zero_path():
    with pytest.raises(ValueError, match="requires delay AdaLN-Zero"):
        SmolVLAConfig(observation_delay_zero_bypass=True)
    with pytest.raises(ValueError, match="without noise AdaRMS"):
        SmolVLAConfig(
            streaming=True,
            observation_delay_max_tokens=4,
            use_observation_delay_adaln_zero=True,
            observation_delay_zero_bypass=True,
            use_noise_adarms=True,
        )


@pytest.mark.parametrize("unit", ["tokens", "chunks"])
def test_sincos_encodes_action_age_without_clipping_or_normalizing_positions(make_model, unit):
    model = make_model(sincos=True, delay_unit=unit)
    key = f"observation_delay_{unit}"
    delay = model._delay_tensor(
        torch.tensor([4]) if unit == "chunks" else None,
        torch.tensor([4]) if unit == "tokens" else None,
        batch_size=1,
        device=torch.device("cpu"),
    )
    features = model._delay_position_encoding(delay, 4)
    base = 8 if unit == "chunks" else 4
    positions = torch.arange(base, base + 4).float()
    torch.testing.assert_close(features[0, :, 0], positions.sin())
    torch.testing.assert_close(features[0, :, 4], positions.cos())
    condition = model._observation_delay_condition(1, torch.device("cpu"), **{key: torch.tensor([4])})
    torch.testing.assert_close(condition, model.observation_delay_adarms(features))
    assert not torch.allclose(condition[:, 0], condition[:, -1])
    # Moving one action forward is identical to moving the observation one step backward.
    shift = 2 if unit == "chunks" else 1
    previous = model._observation_delay_condition(1, torch.device("cpu"), **{key: torch.tensor([3])})
    torch.testing.assert_close(previous[:, shift:], condition[:, :-shift])


@pytest.mark.parametrize("route", ["embedding", "adarms", "zero", "token", "mixed"])
@pytest.mark.parametrize("attention_mode", ["self_attn", "cross_attn"])
@pytest.mark.parametrize("unit", ["tokens", "chunks"])
@pytest.mark.parametrize("tactile", [False, True])
def test_sincos_training_cache_parity_and_condition_layout(make_model, route, attention_mode, unit, tactile):
    torch.manual_seed(19)
    adaptive = route in ("adarms", "zero", "mixed")
    token = route in ("token", "mixed")
    model = make_model(
        attention_mode=attention_mode,
        enabled=adaptive,
        adaln_zero=route == "zero",
        sincos=True,
        embedding=route in ("embedding", "mixed"),
        delay_token=token,
        delay_unit=unit,
        tactile=tactile,
    ).eval()
    for module in model.modules():
        if isinstance(module, (AdaptiveRMSNormModulation, AdaptiveRMSNormZeroModulation)):
            nn.init.normal_(module.projection.weight, std=0.1)
    tactile_emb = torch.randn(2, model.vlm_with_expert.expert_hidden_size) if tactile else None
    # Keep this check focused on timing/layout; the tactile encoder has separate coverage.
    model._encode_tactile = lambda batch: tactile_emb
    tactile_batch = {} if tactile else None
    key = f"observation_delay_{unit}"
    delays = {key: torch.tensor([0, 4])}
    state = torch.randn(2, model.config.state_dim)
    lang = torch.tensor([[1, 2], [3, 4]])
    lang_mask = torch.ones_like(lang, dtype=torch.bool)
    actions = torch.randn(2, 4, model.config.action_dim)
    noise = torch.randn_like(actions)
    time = torch.tensor([[0.2, 0.2, 0.8, 0.8]]).expand(2, -1)
    x_t = time[..., None] * noise + (1 - time[..., None]) * actions
    cache = model.encode_prefix([], [], lang, lang_mask, state, **{key: torch.zeros(2, dtype=torch.long)})
    velocity = model.denoise_step(
        cache["prefix_pad_masks"],
        cache["past_key_values"],
        x_t,
        time,
        tactile_batch=tactile_batch,
        apply_head_guidance=False,
        **delays,
    )
    loss = model(
        [],
        [],
        lang,
        lang_mask,
        state,
        actions,
        noise=noise,
        time=time,
        tactile_batch=tactile_batch,
        **delays,
    )
    expected = (noise - actions - velocity).square()
    if tactile:
        _, copy = model.denoise_step(
            cache["prefix_pad_masks"],
            cache["past_key_values"],
            x_t,
            time,
            tactile_batch=tactile_batch,
            apply_head_guidance=False,
            return_tactile_copy=True,
            **delays,
        )
        expected[:, 0] = 0.5 * (expected[:, 0] + (noise[:, 0] - actions[:, 0] - copy).square())
    torch.testing.assert_close(loss, expected)
    condition = model._observation_delay_condition(
        2,
        state.device,
        tactile_token=tactile,
        **delays,
    )
    suffix, _, _ = model.embed_suffix(x_t, time, tactile_embedding=tactile_emb, **delays)
    assert suffix.shape[1] == 4 + (4 if token else 0) + (2 if tactile else 0)
    if adaptive:
        assert condition.shape == suffix.shape
        if token:
            torch.testing.assert_close(condition[:, :4], condition[:, -4:])
        if tactile:
            torch.testing.assert_close(condition[:, -6:-4], condition[:, -4:-3].expand(-1, 2, -1))
    if token:
        # Delay conditions leave the noisy action/time features untouched before attention.
        zero_suffix, _, _ = model.embed_suffix(
            x_t,
            time,
            tactile_embedding=tactile_emb,
            **{key: torch.zeros(2, dtype=torch.long)},
        )
        if route == "token":
            torch.testing.assert_close(suffix[:, -4:], zero_suffix[:, -4:])
        assert not torch.allclose(suffix[1, :4], zero_suffix[1, :4])
    loss.mean().backward()
    encoders = [
        model.observation_delay_embedding,
        model.observation_delay_adarms,
        model.observation_delay_token_embedding,
    ]
    for encoder in encoders:
        if encoder is not None:
            assert encoder[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["diffusion", "stream", "stream-head"])
@pytest.mark.parametrize("route", ["embedding", "adarms", "zero", "token"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sincos_sampling_entrypoints_refresh_per_action_ages(make_model, mode, route, dtype):
    model = (
        make_model(
            enabled=route in ("adarms", "zero"),
            adaln_zero=route == "zero",
            sincos=True,
            embedding=route == "embedding",
            delay_token=route == "token",
        )
        .to(dtype)
        .eval()
    )
    model.reset_stream_head_state()  # The public policy resets streaming state before inference.
    cache = model.encode_prefix(
        [],
        [],
        torch.tensor([[1, 2]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.randn(1, model.config.state_dim).to(dtype),
        observation_delay_tokens=torch.tensor([0]),
    )
    encoder = next(
        module
        for module in (
            model.observation_delay_embedding,
            model.observation_delay_adarms,
            model.observation_delay_token_embedding,
        )
        if module is not None
    )
    received = []
    handle = encoder[0].register_forward_pre_hook(lambda module, args: received.append(args[0].clone()))
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            for delay in (0, 4):
                received.clear()
                actions = model.sample_actions_from_prefix(
                    cache,
                    inference_mode=mode,
                    observation_delay_tokens=torch.tensor([delay]),
                )
                assert actions.shape == (1, model.config.chunk_size, model.config.action_dim)
                assert torch.isfinite(actions).all()
                assert received
                expected = model._delay_position_encoding(torch.tensor([delay]), model.config.chunk_size).to(
                    dtype
                )
                for features in received:
                    torch.testing.assert_close(features, expected)
    finally:
        handle.remove()


@pytest.mark.parametrize("per_token", [False, True])
def test_noise_embedding_switch_controls_only_explicit_timestep_condition(make_model, per_token):
    torch.manual_seed(9)
    model = make_model()
    x_t = torch.randn(2, 4, model.config.action_dim)
    time = torch.full((2, 4) if per_token else (2,), 0.2)
    first, _, _ = model.embed_suffix(x_t, time)
    second, _, _ = model.embed_suffix(x_t, time + 0.5)
    assert not torch.allclose(first, second)

    disabled = make_model(use_noise_embedding=False)
    disabled.load_state_dict(model.state_dict(), strict=True)
    first, _, _ = disabled.embed_suffix(x_t, time)
    second, _, _ = disabled.embed_suffix(x_t, time + 0.5)
    torch.testing.assert_close(first, second)
    changed_actions, _, _ = disabled.embed_suffix(x_t + 1, time)
    assert not torch.allclose(first, changed_actions)

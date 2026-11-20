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

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.policies.smolvla.smolvlm_with_expert import FlowTimeRMSNorm
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from tests.policies.smolvla import test_delay_adarms

make_model = test_delay_adarms.make_model


@pytest.fixture
def gate_model(make_model):
    torch.manual_seed(4)
    template = make_model(tactile=True, use_noise_adarms=True, use_noise_embedding=False)
    model = VLAFlowMatching(
        replace(template.config, use_tactile_cfg_gate=True, tactile_cfg_gate_hidden_dim=8)
    )
    # Simulate a trained backbone: fresh AdaRMS residual gates otherwise stay
    # zero and CA0 is identical to A0, yielding no signal for the CFG gate.
    for module in model.modules():
        if isinstance(module, FlowTimeRMSNorm):
            nn.init.normal_(module.projection.weight, std=0.2)
    return model


def tactile_batch():
    return {f"observation.tactile.{side}_marker": torch.randn(2, 1, 2, 3, 2) for side in ("left", "right")}


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_tactile_token": False},
        {"tactile_cfg_gate_hidden_dim": 0},
        {"tactile_cfg_lambda_init": 0.0},
        {"tactile_cfg_lambda_init": 2.0},
        {"tactile_cfg_lambda_max": float("inf")},
    ],
)
def test_gate_config_validation(overrides):
    config = {"use_tactile": True, "use_tactile_token": True, "use_tactile_cfg_gate": True}
    config.update(overrides)
    with pytest.raises(ValueError):
        SmolVLAConfig(**config)


def test_gate_initialization_and_final_t_selection(gate_model):
    model = gate_model
    suffix = torch.randn(2, 9, 8, requires_grad=True)  # Leading delay tokens must not change T selection.
    strength = model._tactile_cfg_strength(suffix, 4)
    torch.testing.assert_close(strength, torch.full((2, 1), 0.5))
    captured = []
    handle = model.tactile_cfg_gate.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    model._tactile_cfg_strength(suffix, 4)
    handle.remove()
    torch.testing.assert_close(captured[0], suffix[:, -6])
    assert not captured[0].requires_grad
    with torch.no_grad():
        model.tactile_cfg_gate[-1].bias.copy_(torch.tensor([-20.0]))
    assert (model._tactile_cfg_strength(suffix, 4) < 0.001).all()
    with torch.no_grad():
        model.tactile_cfg_gate[-1].bias.fill_(20.0)
    strength = model._tactile_cfg_strength(suffix, 4)
    assert ((strength > 1.999) & (strength <= 2.0)).all()


def test_gate_fusion_loss_updates_only_gate_and_matches_cached_inference(gate_model):
    torch.manual_seed(12)
    model = gate_model
    model.config.tactile_token_history_length = 3
    batch = tactile_batch()
    batch = {key: value.expand(-1, 3, -1, -1, -1).clone() for key, value in batch.items()}
    model._encode_tactile(batch)  # Materialize LazyLinear before snapshotting.
    model.train()
    assert not model.training
    assert model.tactile_cfg_gate.training
    assert all(
        not module.training
        for name, module in model.named_modules()
        if name and not name.startswith("tactile_cfg_gate")
    )
    assert all(
        p.requires_grad == name.startswith("tactile_cfg_gate.") for name, p in model.named_parameters()
    )
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    lang = torch.tensor([[1, 2], [3, 4]])
    mask = torch.ones_like(lang, dtype=torch.bool)
    state = torch.randn(2, model.config.state_dim)
    actions = torch.randn(2, 4, model.config.action_dim)
    noise = torch.randn_like(actions)
    time = torch.tensor([[0.2, 0.2, 0.8, 0.8]]).expand(2, -1)
    x_t = time[..., None] * noise + (1 - time[..., None]) * actions
    cache = model.encode_prefix([], [], lang, mask, state)
    raw, tactile, strength = model.denoise_step(
        cache["prefix_pad_masks"],
        cache["past_key_values"],
        x_t,
        time,
        tactile_batch=batch,
        return_tactile_copy=True,
        return_tactile_lambda=True,
        apply_head_guidance=False,
    )
    expected = ((raw[:, 0] + strength * (tactile - raw[:, 0]) - (noise - actions)[:, 0]) ** 2)[:, None]

    def loss():
        return model([], [], lang, mask, state, actions, noise, time, tactile_batch=batch)

    initial = loss()
    assert initial.shape == (2, 1, model.config.action_dim)
    torch.testing.assert_close(initial, expected)
    optimizer = torch.optim.Adam(model.tactile_cfg_gate.parameters(), lr=0.01)
    for _ in range(8):
        optimizer.zero_grad()
        current = loss().mean()
        current.backward()
        assert model.tactile_cfg_gate[-1].bias.grad.abs().sum() > 0
        assert all(p.grad is None for name, p in model.named_parameters() if name in frozen)
        optimizer.step()
    assert loss().mean() < initial.mean()
    for name, p in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
    model.eval()
    assert not model.tactile_cfg_gate.training


@pytest.mark.parametrize("action_width", [3, 6])
def test_gate_policy_reduction_uses_only_first_action_padding(gate_model, action_width):
    model = gate_model
    config = model.config
    config.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,))}
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_width,))}
    policy = object.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = config
    policy.model = model
    policy._prepare_batch = lambda batch, **kwargs: batch
    policy.prepare_images = lambda batch: ([], [])
    batch = {
        **tactile_batch(),
        OBS_STATE: torch.randn(2, 6),
        ACTION: torch.randn(2, 4, 6),
        OBS_LANGUAGE_TOKENS: torch.tensor([[1, 2], [3, 4]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 2, dtype=torch.bool),
        "action_is_pad": torch.tensor([[False, True, True, True], [True, False, False, False]]),
    }
    noise = torch.randn(2, 4, 6)
    time = torch.tensor([0.3, 0.6])
    per_sample, metrics = policy(batch, noise=noise, time=time, reduction="none")
    mean_loss, _ = policy(batch, noise=noise, time=time)
    assert per_sample[0] > 0 and per_sample[1] == 0
    torch.testing.assert_close(mean_loss, per_sample[0])
    assert metrics["tactile_cfg_lambda_mean"] == pytest.approx(0.5)
    errors = model._last_tactile_training_errors
    for name in ("a0_target_mse", "ca0_target_mse", "ca0_a0_mse", "fused_target_mse"):
        assert metrics[f"tactile/{name}"] == pytest.approx(errors[name][0, :action_width].mean().item())
        assert not errors[name].requires_grad
    assert metrics["tactile/ca0_a0_rms"] == pytest.approx(
        errors["ca0_a0_mse"][0, :action_width].mean().sqrt().item()
    )
    assert metrics["tactile/fused_target_mse"] == pytest.approx(mean_loss.item())
    assert metrics["tactile/ca0_target_improvement"] == pytest.approx(
        metrics["tactile/a0_target_mse"] - metrics["tactile/ca0_target_mse"], abs=1e-6
    )
    assert metrics["tactile/ca0_better_fraction"] == float(
        metrics["tactile/ca0_target_mse"] < metrics["tactile/a0_target_mse"]
    )
    assert set(map(id, policy.get_optim_params())) == set(map(id, model.tactile_cfg_gate.parameters()))


@pytest.mark.parametrize("missing_backbone", [False, True])
def test_load_tactile_checkpoint_initializes_only_missing_gate(gate_model, tmp_path, missing_backbone):
    model = gate_model
    model._encode_tactile(tactile_batch())
    policy = object.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = model.config
    policy.model = model
    weights = {
        name: value
        for name, value in policy.state_dict().items()
        if not name.startswith("model.tactile_cfg_gate.")
    }
    if missing_backbone:
        del weights["model.tactile_encoder.2.weight"]
    path = str(tmp_path / "model.safetensors")
    save_file(weights, path)
    if missing_backbone:
        with pytest.raises(ValueError, match="complete tactile-token checkpoint"):
            SmolVLAPolicy._load_as_safetensor(policy, path, "cpu", False)
    else:
        SmolVLAPolicy._load_as_safetensor(policy, path, "cpu", False)
        assert model._tactile_cfg_lambda_logger.checkpoint == tmp_path.resolve()
        assert model._tactile_cfg_lambda_logger.path is None
        torch.testing.assert_close(
            model._tactile_cfg_strength(torch.randn(2, 6, 8), 4), torch.full((2, 1), 0.5)
        )
        # Gate-bearing checkpoints must preserve learned weights rather than reinitializing.
        with torch.no_grad():
            model.tactile_cfg_gate[-1].bias.fill_(0.7)
        saved_gate = model.tactile_cfg_gate[-1].bias.detach().clone()
        save_file(policy.state_dict(), path)
        with torch.no_grad():
            model.tactile_cfg_gate[-1].bias.zero_()
        SmolVLAPolicy._load_as_safetensor(policy, path, "cpu", False)
        torch.testing.assert_close(model.tactile_cfg_gate[-1].bias, saved_gate)


@pytest.mark.parametrize(
    "mode", ["stream-head-tactile-cfg", "stream-head-mean-tactile-cfg", "stream-head-tactile"]
)
def test_stream_head_uses_per_sample_gate_only_for_cfg(mode):
    model = object.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        use_tactile_cfg_gate=True,
        use_tactile_token=True,
        chunk_size=2,
        streaming=True,
        streaming_chunk_size=2,
        action_dim=1,
    )
    model.tactile_adarms = None
    model._encode_tactile = lambda batch: torch.ones(2, 1)
    raw = torch.full((2, 2, 1), 2.0)
    tactile = torch.full((2, 1), 10.0)
    strength = torch.tensor([[0.25], [1.75]])

    def denoise(**kwargs):
        if mode == "stream-head-tactile":
            assert not kwargs.get("return_tactile_lambda", False)
            return raw, tactile
        assert kwargs["return_tactile_lambda"]
        return raw, tactile, strength

    model.denoise_step = denoise
    model.sample_noise = lambda shape, device: torch.zeros(shape)
    model._stream_head_state = {
        "action_window": torch.zeros(2, 2, 1),
        "timestep_window": torch.ones(2, 2),
        "history": torch.full((2, 2, 1, 1), 6.0),
        "history_valid": torch.ones(2, 2, 1, dtype=torch.bool),
    }
    prefix = {"prefix_pad_masks": torch.ones(2, 1, dtype=torch.bool), "past_key_values": None}
    actions = model.stream_head_step_from_prefix(
        prefix, tactile_batch={}, inference_mode=mode, tactile_cfg_lambda=99.0
    )
    base = 4.0 if mode == "stream-head-mean-tactile-cfg" else 2.0
    expected = -tactile if mode == "stream-head-tactile" else -(base + strength * 8.0)
    torch.testing.assert_close(actions[:, 0], expected)
    torch.testing.assert_close(actions[:, 1], torch.zeros(2, 1))
    torch.testing.assert_close(model._stream_head_state["history"][:, :, -1], raw)


def test_random_tactile_changes_ca0_but_preserves_raw_fields_and_action_rng(gate_model):
    model = gate_model
    policy = object.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = model.config
    policy.config.use_tactile_token_history = True
    policy.config.tactile_token_history_length = 3
    policy.model = model
    policy.reset()
    policy.eval()
    policy.enable_random_tactile_experiment()
    batch = {**tactile_batch(), OBS_STATE: torch.zeros(2, 6)}
    original = {k: v.clone() for k, v in batch.items()}
    rng = torch.get_rng_state().clone()
    randomized = policy._prepare_batch(batch)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    for side in ("left", "right"):
        key = f"observation.tactile.{side}_marker"
        assert not torch.equal(randomized[key], batch[key])
        torch.testing.assert_close(batch[key], original[key])
        history = randomized[f"{key}_history"]
        torch.testing.assert_close(history[:, -1], randomized[key])
    next_frame = policy._prepare_batch(batch)
    key = "observation.tactile.left_marker"
    assert not torch.equal(next_frame[key], randomized[key])
    torch.testing.assert_close(next_frame[f"{key}_history"][:, -2], randomized[key])
    policy.reset()
    replay = policy._prepare_batch(batch)
    torch.testing.assert_close(replay[key], randomized[key])
    # Prefix prefill must never sample a random frame or advance history.
    policy.reset()
    untouched = policy._prepare_batch(batch, update_tactile_history=False)
    assert not policy._tactile_history
    torch.testing.assert_close(untouched[key], batch[key])
    assert not policy._random_tactile_generators

    lang = torch.tensor([[1, 2], [3, 4]])
    mask = torch.ones_like(lang, dtype=torch.bool)
    cache = model.encode_prefix([], [], lang, mask, batch[OBS_STATE])
    x_t = torch.randn(2, 4, 6)
    time = torch.full((2, 4), 0.1)
    results = []
    with torch.no_grad():
        for observation in (batch, randomized):
            results.append(
                model.denoise_step(
                    cache["prefix_pad_masks"],
                    cache["past_key_values"],
                    x_t,
                    time,
                    tactile_batch=observation,
                    return_tactile_copy=True,
                    apply_head_guidance=False,
                )
            )
    # Paired observations, identical prefix/noise/time: only CA0 consumes tactile.
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    assert not torch.allclose(results[0][1], results[1][1])
    # Even when enabled, the experimental flag does not alter training inputs.
    policy.reset()
    policy.train()
    training = policy._prepare_batch(batch)
    torch.testing.assert_close(training[key], batch[key])
    assert not policy._random_tactile_generators


def test_random_tactile_requires_token_policy(gate_model):
    policy = object.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = replace(gate_model.config, use_tactile_token=False, use_tactile_cfg_gate=False)
    policy.model = gate_model
    with pytest.raises(ValueError, match="requires use_tactile_token"):
        policy.enable_random_tactile_experiment()

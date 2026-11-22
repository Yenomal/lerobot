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

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from tests.policies.smolvla import test_delay_adarms
from tests.policies.smolvla.test_tactile_cfg_gate import tactile_batch

make_model = test_delay_adarms.make_model


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_fm_tactile_token": False},
        {"use_tactile_cfg_gate": True},
        {"use_peft": True},
        {"tactile_reconstruction_hidden_dim": 0},
    ],
)
def test_reconstruction_config_validation(overrides):
    config = {"use_tactile": True, "use_fm_tactile_token": True, "use_tactile_reconstruction": True}
    config.update(overrides)
    with pytest.raises(ValueError):
        SmolVLAConfig(**config)


@pytest.mark.parametrize("history", [False, True])
def test_reconstruction_target_gradients_reduction_and_checkpoint(make_model, tmp_path, history, monkeypatch):
    torch.manual_seed(4)
    template = make_model(tactile=True, delay_token=True, expert_width_multiplier=2.0)
    model = VLAFlowMatching(
        replace(
            template.config,
            use_tactile_reconstruction=True,
            tactile_reconstruction_hidden_dim=16,
            use_tactile_tcn=history,
            tactile_token_history_length=(5 if history else 3),
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
                **{
                    f"observation.tactile.{side}_marker": PolicyFeature(
                        type=FeatureType.STATE, shape=(2, 3, 2)
                    )
                    for side in ("left", "right")
                },
            },
        )
    )
    config = model.config
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))}
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
    for side in ("left", "right"):
        key = f"observation.tactile.{side}_marker"
        batch[key] = torch.cat([torch.randn_like(batch[key])] * (4 if history else 2) + [batch[key]], dim=1)
    target = torch.cat(
        [batch[f"observation.tactile.{side}_marker"][:, -1].flatten(1) for side in ("left", "right")], dim=-1
    )
    assert target.shape == (2, 24)  # Marker width differs from FM (16) and action (6).
    embedding = model._encode_tactile(batch).detach()
    if history:
        changed_history = {key: value.clone() for key, value in batch.items()}
        for side in ("left", "right"):
            changed_history[f"observation.tactile.{side}_marker"][:, 0] += 10
        # Earlier history affects the probe input, while the target stays current-frame markers.
        assert not torch.allclose(model._encode_tactile(changed_history), embedding)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    policy.train()
    assert policy.training and not model.training and model.tactile_reconstruction.training
    assert all(
        p.requires_grad == name.startswith("tactile_reconstruction.") for name, p in model.named_parameters()
    )
    assert set(map(id, policy.get_optim_params())) == set(map(id, model.tactile_reconstruction.parameters()))

    def forbidden_forward(*args, **kwargs):
        raise AssertionError("Tactile encoder probe must not run VLM/FM")

    monkeypatch.setattr(model.vlm_with_expert, "forward", forbidden_forward)
    noise = torch.randn(2, 4, 6)
    time = torch.tensor([0.3, 0.6])
    per_sample, metrics = policy(batch, noise=noise, time=time, reduction="none")
    probe_input = embedding[:, -1] if embedding.ndim == 3 else embedding
    expected = (model.tactile_reconstruction(probe_input) - target).square().mean(dim=-1)
    torch.testing.assert_close(per_sample[0], expected[0])
    assert per_sample[1] == 0
    assert "tactile_cfg_lambda_mean" not in metrics
    initial, _ = policy(batch, noise=noise, time=time)
    torch.testing.assert_close(initial, expected[0])
    optimizer = torch.optim.Adam(policy.get_optim_params(), lr=0.01)
    for _ in range(8):
        optimizer.zero_grad()
        loss, _ = policy(batch, noise=noise, time=time)
        loss.backward()
        assert model.tactile_reconstruction[-1].weight.grad.abs().sum() > 0
        assert all(p.grad is None for name, p in model.named_parameters() if name in frozen)
        optimizer.step()
    final, _ = policy(batch, noise=noise, time=time)
    assert final < initial
    for name, p in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
    policy.eval()
    assert not model.tactile_reconstruction.training

    path = str(tmp_path / "model.safetensors")
    saved = {name: value.clone() for name, value in policy.state_dict().items()}
    save_file(saved, path)
    SmolVLAPolicy._load_as_safetensor(policy, path, "cpu", False)
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, saved[name])
    backbone = {
        name: value for name, value in saved.items() if not name.startswith("model.tactile_reconstruction.")
    }
    save_file(backbone, path)
    SmolVLAPolicy._load_as_safetensor(policy, path, "cpu", False)
    del backbone["model.tactile_encoder.2.weight"]
    save_file(backbone, path)
    with pytest.raises(ValueError, match="complete tactile-token checkpoint"):
        SmolVLAPolicy._load_as_safetensor(policy, path, "cpu", False)

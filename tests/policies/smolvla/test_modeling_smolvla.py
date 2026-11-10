#!/usr/bin/env python

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
from safetensors.torch import save_file

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE


@pytest.mark.parametrize("resize", [(4, 4), None])
def test_prepare_images_normalizes_uint8_inputs_before_resizing(resize):
    policy = SimpleNamespace(
        config=SimpleNamespace(
            image_features={"observation.images.camera": object()},
            resize_imgs_with_padding=resize,
            empty_cameras=0,
        )
    )
    image = torch.tensor([[[[0, 255], [128, 64]]]], dtype=torch.uint8).repeat(1, 3, 1, 1)

    images, image_masks = SmolVLAPolicy.prepare_images(policy, {"observation.images.camera": image})

    assert images[0].dtype == torch.float32
    assert images[0].min() >= -1.0
    assert images[0].max() <= 1.0
    assert image_masks[0].tolist() == [True]
    if resize is None:
        torch.testing.assert_close(images[0], image.float() / 255.0 * 2.0 - 1.0)


def test_smolvla_dimensions_default_to_six_and_do_not_hide_feature_mismatch():
    config = SmolVLAConfig()
    assert config.state_dim == 6
    assert config.action_dim == 6

    config.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(9,))}
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(9,))}
    assert config.state_dim == 6
    assert config.action_dim == 6
    with pytest.raises(ValueError, match="Set --policy.state_dim"):
        config.validate_features()


def test_pretrained_config_validates_state_width_against_runtime_dataset():
    config = SmolVLAConfig(state_dim=9, action_dim=9)
    config.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,))}
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(9,))}
    config._runtime_dataset_meta = SimpleNamespace(features={OBS_STATE: {"shape": (9,)}})

    config.validate_features()


def test_prepare_state_and_action_use_explicit_dynamic_dimensions():
    policy = SimpleNamespace(config=SimpleNamespace(state_dim=9, action_dim=9))
    state = torch.randn(2, 9)
    action = torch.randn(2, 50, 9)

    assert SmolVLAPolicy.prepare_state(policy, {OBS_STATE: state}).shape == (2, 9)
    assert SmolVLAPolicy.prepare_action(policy, {ACTION: action}).shape == (2, 50, 9)


def test_prepare_state_and_action_reject_implicit_truncation():
    policy = SimpleNamespace(config=SimpleNamespace(state_dim=6, action_dim=6))
    with pytest.raises(ValueError, match="state_dim=6"):
        SmolVLAPolicy.prepare_state(policy, {OBS_STATE: torch.randn(2, 9)})
    with pytest.raises(ValueError, match="action_dim=6"):
        SmolVLAPolicy.prepare_action(policy, {ACTION: torch.randn(2, 50, 9)})


def test_pretrained_dimension_adaptation_overlap_copies_projection_weights(tmp_path):
    class ProjectionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                adapt_pretrained_dimensions=True,
                _pretrained_projection_dims={"state": 6, "action": 6},
            )
            self.model = torch.nn.Module()
            self.model.state_proj = torch.nn.Linear(9, 4)
            self.model.action_in_proj = torch.nn.Linear(9, 5)
            self.model.action_out_proj = torch.nn.Linear(5, 9)

    model = ProjectionModel()
    initial_state_projection = model.model.state_proj.weight.detach().clone()
    initial_action_input = model.model.action_in_proj.weight.detach().clone()
    initial_action_output = model.model.action_out_proj.weight.detach().clone()
    initial_action_output_bias = model.model.action_out_proj.bias.detach().clone()
    checkpoint = {
        "model.state_proj.weight": torch.arange(4 * 32, dtype=torch.float32).reshape(4, 32),
        "model.action_in_proj.weight": torch.arange(5 * 32, dtype=torch.float32).reshape(5, 32),
        "model.action_out_proj.weight": torch.arange(32 * 5, dtype=torch.float32).reshape(32, 5),
        "model.action_out_proj.bias": torch.arange(32, dtype=torch.float32),
    }
    checkpoint_path = tmp_path / "model.safetensors"
    save_file(checkpoint, str(checkpoint_path))

    SmolVLAPolicy._load_as_safetensor(model, str(checkpoint_path), "cpu", False)

    torch.testing.assert_close(model.model.state_proj.weight[:, :6], checkpoint["model.state_proj.weight"][:, :6])
    torch.testing.assert_close(model.model.action_in_proj.weight[:, :6], checkpoint["model.action_in_proj.weight"][:, :6])
    torch.testing.assert_close(
        model.model.action_out_proj.weight[:6, :], checkpoint["model.action_out_proj.weight"][:6, :]
    )
    torch.testing.assert_close(model.model.action_out_proj.bias[:6], checkpoint["model.action_out_proj.bias"][:6])
    torch.testing.assert_close(model.model.state_proj.weight[:, 6:], initial_state_projection[:, 6:])
    torch.testing.assert_close(model.model.action_in_proj.weight[:, 6:], initial_action_input[:, 6:])
    torch.testing.assert_close(model.model.action_out_proj.weight[6:, :], initial_action_output[6:, :])
    torch.testing.assert_close(model.model.action_out_proj.bias[6:], initial_action_output_bias[6:])

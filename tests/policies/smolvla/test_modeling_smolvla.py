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
from types import MethodType

import pytest
import torch
from safetensors.torch import save_file

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
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


def test_smolvla_adopts_dataset_state_and_visual_features_only():
    config = SmolVLAConfig()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.camera3": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    }
    config.set_dataset_feature_metadata(
        {
            OBS_STATE: {"dtype": "float32", "shape": (9,), "names": None},
            "observation.images.camera1": {
                "dtype": "video",
                "shape": (3, 270, 480),
                "names": ["channels", "height", "width"],
            },
            "observation.images.camera2": {
                "dtype": "video",
                "shape": (3, 270, 480),
                "names": ["channels", "height", "width"],
            },
            "observation.tactile.left_marker": {
                "dtype": "float32",
                "shape": (2, 63, 2),
                "names": None,
            },
        }
    )

    assert config.input_features == {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(9,)),
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 270, 480)),
        "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 270, 480)),
    }


@pytest.mark.parametrize("mode", ["bidirectional", "mask", "causal", "random_gate"])
def test_streaming_attention_modes_are_supported(mode):
    config = SmolVLAConfig(
        streaming=True,
        chunk_size=10,
        n_action_steps=10,
        streaming_chunk_size=5,
        streaming_attention_mode=mode,
    )
    model = object.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = config
    boundaries = model._action_att_masks(2, torch.device("cpu"), torch.bool)

    assert boundaries.shape == (2, 10)
    assert boundaries[:, 0].all()
    if mode in ("bidirectional", "mask"):
        assert boundaries[:, 1:].sum() == 0
    elif mode == "random_gate":
        assert boundaries[:, 1:].sum() == 0
    elif mode == "causal":
        assert boundaries[:, 5].all()


def test_random_gate_uses_log_uniform_soft_biases():
    config = SmolVLAConfig(
        streaming=True,
        chunk_size=10,
        n_action_steps=10,
        streaming_chunk_size=5,
        streaming_attention_mode="random_gate",
    )
    model = object.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = config

    log_gates = model._random_attention_log_gates(2, torch.device("cpu"), torch.float32)
    assert log_gates.shape == (2, 2)
    assert torch.all(log_gates <= 1.0)
    assert torch.all(log_gates >= torch.log(torch.tensor(torch.finfo(torch.float32).tiny)))

    bias = model._make_action_attention_bias(
        log_gates,
        query_length=12,
        key_length=12,
        action_query_start=2,
        action_key_start=2,
        action_count=10,
    )
    assert torch.allclose(bias[:, 2:7, 2:7], torch.zeros(2, 5, 5))
    assert torch.allclose(bias[:, 7:, 7:], torch.zeros(2, 5, 5))
    assert torch.allclose(bias[:, 2:7, 7:], log_gates[:, :1, None].expand(2, 5, 5))


def test_stream_head_updates_every_chunk_head():
    velocity = torch.arange(1.0, 11.0).reshape(1, 10, 1)
    guided = VLAFlowMatching._apply_head_guidance(
        velocity,
        inference_mode="stream-head",
        context_cfg_lambda=0.1,
        chunk_size=10,
        streaming_chunk_size=5,
    )
    assert guided[0, :, 0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0])


def test_stream_head_mean_uses_same_relative_token_across_chunks():
    velocity = torch.arange(1.0, 11.0).reshape(1, 10, 1)
    guided = VLAFlowMatching._apply_head_guidance(
        velocity,
        inference_mode="stream-head-mean",
        context_cfg_lambda=0.1,
        chunk_size=5,
        streaming_chunk_size=5,
    )
    assert guided[0, 0, 0].item() == pytest.approx(3.5)
    assert guided[0, 5, 0].item() == pytest.approx(3.5)
    assert guided[0, 1:5].abs().sum().item() == pytest.approx(0.0)

    guided_cfg = VLAFlowMatching._apply_head_guidance(
        velocity,
        inference_mode="stream-head-context-cfg",
        context_cfg_lambda=0.2,
        chunk_size=5,
        streaming_chunk_size=5,
    )
    assert guided_cfg[0, 0, 0].item() == pytest.approx(1.0 + 0.2 * ((2 + 3 + 4 + 5) - 4 * 1))


def test_streaming_delay_and_tactile_config_validation():
    with pytest.raises(ValueError, match="requires streaming=True"):
        SmolVLAConfig(observation_delay_max_chunks=1)
    with pytest.raises(ValueError, match="cannot both be non-zero"):
        SmolVLAConfig(streaming=True, observation_delay_max_chunks=1, observation_delay_max_tokens=1)
    with pytest.raises(ValueError, match="streaming_attention_mode"):
        SmolVLAConfig(streaming_attention_mode="casual")
    with pytest.raises(ValueError, match="streaming_attention_mode"):
        SmolVLAConfig(streaming_attention_mode="ramdom_gate")
    with pytest.raises(ValueError, match="requires use_tactile=True"):
        SmolVLAConfig(use_tactile_token=True)


def test_observation_delay_selects_history_frame_before_training_forward():
    config = SimpleNamespace(
        adapt_to_pi_aloha=False,
        observation_delay_max_chunks=0,
        observation_delay_max_tokens=2,
        streaming_chunk_size=5,
        image_features={"observation.images.camera": object()},
        use_tactile=False,
        use_tactile_token_history=False,
    )
    policy = object.__new__(SmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    batch = {
        OBS_STATE: torch.tensor([[[10.0], [20.0], [30.0]]]),
        "observation.images.camera": torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3, 1, 1, 1),
        "observation_delay_tokens": torch.tensor([1]),
    }

    prepared = policy._prepare_batch(batch, update_tactile_history=False)

    assert prepared[OBS_STATE].tolist() == [[20.0]]
    assert prepared["observation.images.camera"].flatten().tolist() == [2.0]
    assert prepared["observation_delay_tokens"].tolist() == [1]

    constant_sample = policy._prepare_batch(
        {
            OBS_STATE: torch.tensor([[[10.0], [20.0], [30.0]]]),
            "observation.images.camera": torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3, 1, 1, 1),
            "observation_delay_tokens": torch.tensor([2]),
        },
        update_tactile_history=False,
        delay_mask=torch.tensor([False]),
    )
    assert constant_sample[OBS_STATE].tolist() == [[30.0]]
    assert constant_sample["observation_delay_tokens"].tolist() == [0]


def test_tactile_features_are_kept_when_enabled():
    config = SmolVLAConfig(use_tactile=True)
    config.set_dataset_feature_metadata(
        {
            OBS_STATE: {"dtype": "float32", "shape": (9,), "names": None},
            "observation.images.camera1": {
                "dtype": "video",
                "shape": (3, 270, 480),
                "names": ["channels", "height", "width"],
            },
            "observation.tactile.left_marker": {"dtype": "float32", "shape": (2, 63, 2), "names": None},
            "observation.tactile.right_marker": {"dtype": "float32", "shape": (2, 63, 2), "names": None},
        }
    )
    assert "observation.tactile.left_marker" in config.input_features
    assert "observation.tactile.right_marker" in config.input_features


def test_tactile_history_aggregates_to_legacy_single_frame_projection_width():
    config = SmolVLAConfig(
        use_tactile=True,
        use_tactile_token=True,
        use_tactile_token_history=True,
        tactile_token_history_length=10,
    )
    model = object.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = config
    model.tactile_encoder = torch.nn.Sequential(torch.nn.Linear(504, 8))
    batch = {
        "observation.tactile.left_marker_history": torch.randn(1, 10, 2, 63, 2),
        "observation.tactile.right_marker_history": torch.randn(1, 10, 2, 63, 2),
        "observation.tactile.left_marker": torch.randn(1, 2, 63, 2),
        "observation.tactile.right_marker": torch.randn(1, 2, 63, 2),
    }
    assert model._encode_tactile(batch).shape == (1, 8)


def test_tactile_token_mask_isolates_t_ca0_and_a0():
    mask = torch.ones(1, 10, 10, dtype=torch.bool)
    masked = VLAFlowMatching._apply_tactile_token_mask(
        mask,
        prefix_len=4,
        action_count=4,
    )[0]
    # T reads prefix + itself.
    assert masked[4, :5].all()
    assert not masked[4, 5:].any()
    # CA0 reads prefix + T + itself.
    assert masked[5, :4].all() and masked[5, 4] and masked[5, 5]
    assert not masked[5, 6:].any()
    # A0 reads prefix + its own key, never T or CA0.
    assert masked[6, :4].all() and masked[6, 6]
    assert not masked[6, 4] and not masked[6, 5]
    # Later actions cannot consume any special key.
    assert not masked[7:, 4:7].any()


def test_tactile_token_history_uses_causal_tcn_on_fm_width():
    config = SmolVLAConfig(
        use_tactile=True,
        use_tactile_token=True,
        use_tactile_token_history=True,
        tactile_token_history_length=3,
        tactile_tcn_width=4,
    )
    model = object.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = config
    model.tactile_encoder = torch.nn.Sequential(torch.nn.Linear(504, 8))
    model.tactile_token_proj = torch.nn.Linear(8, 6)
    model.tactile_token_tcn_in = torch.nn.Linear(6, 4)
    model.tactile_token_tcn_1 = torch.nn.Linear(12, 4)
    model.tactile_token_tcn_2 = torch.nn.Linear(12, 4)
    model.tactile_token_out = torch.nn.Linear(4, 6)
    batch = {
        "observation.tactile.left_marker_history": torch.randn(1, 3, 2, 63, 2),
        "observation.tactile.right_marker_history": torch.randn(1, 3, 2, 63, 2),
    }
    token = model._encode_tactile(batch)
    assert token.shape == (1, 6)


def test_stream_head_history_follows_action_token_through_window():
    config = SimpleNamespace(
        chunk_size=5,
        streaming=True,
        streaming_chunk_size=5,
        action_dim=1,
        num_steps=1,
    )
    model = object.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = config
    model._stream_head_state = None

    def fake_denoise(self, *, x_t, **kwargs):
        del kwargs
        return torch.arange(x_t.shape[1], device=x_t.device, dtype=x_t.dtype)[None, :, None].expand_as(x_t)

    model.denoise_step = MethodType(fake_denoise, model)
    model.sample_noise = lambda shape, device: torch.zeros(shape, device=device)
    prefix = {
        "prefix_pad_masks": torch.ones(1, 1, dtype=torch.bool),
        "past_key_values": None,
    }

    # After four shifts, the token that started at v4 has recorded v4, v3,
    # v2 and v1. At v0, stream-head-mean includes the current v0 field.
    for _ in range(5):
        result = model.stream_head_step_from_prefix(
            prefix,
            inference_mode="stream-head-mean",
        )
    assert result.shape == (1, 5, 1)
    assert result[0, 0, 0].item() == pytest.approx(-2.0)


def test_stream_uses_master_style_future_window_without_ready_chunk(monkeypatch):
    config = SimpleNamespace(
        chunk_size=10,
        streaming=True,
        streaming_chunk_size=5,
        action_dim=1,
        num_steps=1,
    )
    model = object.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = config
    model._stream_state = None
    seen_lengths = []

    def fake_integrate(function, noise, num_steps):
        del function, num_steps
        return noise + 1.0

    monkeypatch.setattr("lerobot.policies.smolvla.modeling_smolvla.euler_integrate", fake_integrate)

    def fake_denoise(self, *, x_t, **kwargs):
        del kwargs
        seen_lengths.append(x_t.shape[1])
        return torch.zeros_like(x_t)

    model.denoise_step = MethodType(fake_denoise, model)
    model.sample_noise = lambda shape, device: torch.zeros(shape, device=device)
    prefix = {
        "prefix_pad_masks": torch.ones(1, 1, dtype=torch.bool),
        "past_key_values": None,
    }

    model.stream_step_from_prefix(prefix)
    assert model._stream_state["action_window"].shape == (1, 15, 1)
    model.stream_step_from_prefix(prefix)
    assert seen_lengths == [10]


def test_streaming_timestep_schedule_repeats_chunk_times():
    config = SimpleNamespace(chunk_size=10, streaming=True, streaming_chunk_size=2)
    model = object.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = config
    schedule = model._streaming_timestep_schedule(device=torch.device("cpu"), dtype=torch.float32)
    torch.testing.assert_close(schedule, torch.tensor([0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0]))


def test_streaming_training_time_samples_inside_chunk_intervals():
    config = SimpleNamespace(
        streaming=True,
        streaming_constant_weight=0.0,
        streaming_chunk_wise_weight=1.0,
        streaming_chunk_size=2,
    )
    model = object.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = config
    model.sample_time = lambda batch_size, device: torch.full(
        (batch_size,), 0.5, dtype=torch.float32, device=device
    )
    sampled = model._sample_training_time(1, 8, torch.device("cpu"))
    torch.testing.assert_close(sampled, torch.tensor([[0.125, 0.125, 0.375, 0.375, 0.625, 0.625, 0.875, 0.875]]))


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
    torch.testing.assert_close(
        model.model.action_in_proj.weight[:, :6],
        checkpoint["model.action_in_proj.weight"][:, :6],
    )
    torch.testing.assert_close(
        model.model.action_out_proj.weight[:6, :], checkpoint["model.action_out_proj.weight"][:6, :]
    )
    torch.testing.assert_close(model.model.action_out_proj.bias[:6], checkpoint["model.action_out_proj.bias"][:6])
    torch.testing.assert_close(model.model.state_proj.weight[:, 6:], initial_state_projection[:, 6:])
    torch.testing.assert_close(model.model.action_in_proj.weight[:, 6:], initial_action_input[:, 6:])
    torch.testing.assert_close(model.model.action_out_proj.weight[6:, :], initial_action_output[6:, :])
    torch.testing.assert_close(model.model.action_out_proj.bias[6:], initial_action_output_bias[6:])

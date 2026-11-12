#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import logging
import math
from collections import deque
from typing import Any, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from safetensors import safe_open
from safetensors.torch import load_file

from lerobot.configs import FeatureType
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.feature_spec import FeatureSpec
from lerobot.utils.import_utils import require_package

from ..common.flow_matching import (
    euler_integrate,
    sample_noise,
    sample_time_beta,
)
from ..common.vla_utils import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad,
)
from ..pretrained import PreTrainedPolicy, RTCActionSelectKwargs
from ..rtc.modeling_rtc import RTCProcessor
from ..utils import (
    log_model_loading_keys,
    populate_queues,
)
from .configuration_smolvla import SmolVLAConfig
from .smolvlm_with_expert import SmolVLMWithExpertModel


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"
    # The runtime contract intentionally consumes one current frame. n_obs_steps
    # controls training history sampling, not the number of runtime frames.
    runtime_ignores_n_obs_steps = True

    def supports_rtc(self) -> bool:
        return True

    def __init__(
        self,
        config: SmolVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        require_package("transformers", extra="smolvla")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def initialize_lazy_parameters(self) -> bool:
        """Materialize tactile LazyLinear parameters before distributed wrapping.

        FSDP2 inspects parameter storage during ``accelerator.prepare`` and cannot
        handle uninitialized parameters. The tactile encoder intentionally uses a
        LazyLinear because the sensor layout is dataset-dependent, so initialize it
        from the resolved dataset feature shapes before the optimizer is built.
        """
        tactile_encoder = self.model.tactile_encoder
        if tactile_encoder is None or not isinstance(tactile_encoder[0], nn.LazyLinear):
            return False
        if not tactile_encoder[0].has_uninitialized_params():
            return False

        device = next(self.parameters()).device
        tactile_batch = {}
        for side in ("left", "right"):
            key = f"observation.tactile.{side}_marker"
            feature = (self.config.input_features or {}).get(key)
            if feature is None:
                raise ValueError(f"Missing resolved tactile feature {key!r} for LazyLinear initialization")
            # Training batches carry the observation-step axis even when
            # n_obs_steps=1; mirror that shape so the lazy raw marker encoder
            # is initialized from one physical tactile frame.
            tactile_batch[key] = torch.zeros((1, 1, *feature.shape), device=device, dtype=torch.float32)

        was_training = self.training
        self.eval()
        with torch.no_grad():
            self.model._encode_tactile(tactile_batch)
        if was_training:
            self.train()
        return True

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):
        """Load weights, optionally adapting the three embodiment projections."""
        initialize_lazy = getattr(model, "initialize_lazy_parameters", None)
        if callable(initialize_lazy):
            initialize_lazy()
        if not model.config.adapt_pretrained_dimensions:
            return super()._load_as_safetensor(model, model_file, map_location, strict)

        own_state = model.state_dict()
        with safe_open(model_file, framework="pt", device="cpu") as checkpoint_file:
            mismatched_keys = [
                key
                for key in checkpoint_file.keys()
                if key in own_state and tuple(checkpoint_file.get_slice(key).get_shape()) != tuple(own_state[key].shape)
            ]
        if not mismatched_keys:
            return super()._load_as_safetensor(model, model_file, map_location, strict)

        # Load on CPU so opting into adaptation does not create a full duplicate
        # of the checkpoint in GPU memory during fine-tuning initialization.
        checkpoint = load_file(model_file, device="cpu")
        projection_axes = {
            "model.state_proj.weight": 1,
            "model.action_in_proj.weight": 1,
            "model.action_out_proj.weight": 0,
            "model.action_out_proj.bias": 0,
        }
        pretrained_projection_dims = getattr(model.config, "_pretrained_projection_dims", {})
        adapted = []
        for key, value in checkpoint.items():
            expected = own_state.get(key)
            if expected is None or expected.shape == value.shape:
                continue
            axis = projection_axes.get(key)
            if axis is None or value.ndim != expected.ndim or value.ndim not in (1, 2):
                raise ValueError(
                    f"Cannot adapt incompatible SmolVLA checkpoint tensor {key}: "
                    f"checkpoint shape {tuple(value.shape)}, model shape {tuple(expected.shape)}."
                )
            other_axis = 1 - axis if value.ndim == 2 else None
            if other_axis is not None and value.shape[other_axis] != expected.shape[other_axis]:
                raise ValueError(
                    f"Cannot adapt SmolVLA projection {key}: non-embodiment dimension differs "
                    f"({tuple(value.shape)} vs {tuple(expected.shape)})."
                )
            adapted_value = expected.detach().cpu().clone()
            copy_dim = min(value.shape[axis], expected.shape[axis])
            active_dim = pretrained_projection_dims.get(
                "state" if key == "model.state_proj.weight" else "action"
            )
            if active_dim is not None:
                copy_dim = min(copy_dim, active_dim)
            if value.ndim == 1:
                adapted_value[:copy_dim] = value[:copy_dim]
            elif axis == 0:
                adapted_value[:copy_dim, :] = value[:copy_dim, :]
            else:
                adapted_value[:, :copy_dim] = value[:, :copy_dim]
            checkpoint[key] = adapted_value
            adapted.append(
                f"{key}: {tuple(value.shape)} -> {tuple(expected.shape)}; copied {copy_dim} active dimensions"
            )

        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        if adapted:
            logging.getLogger(__name__).warning(
                "Adapted pretrained SmolVLA projection tensors by overlap-copying: %s", "; ".join(adapted)
            )
        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(f"Missing keys: {missing_keys}; unexpected keys: {unexpected_keys}")
        log_model_loading_keys(missing_keys, unexpected_keys)
        return model

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self._tactile_history = {}
        model_value = getattr(self, "model", None)
        if model_value is not None:
            model_value.reset_stream_head_state()

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(
        self,
        batch: dict[str, Tensor],
        *,
        noise: Tensor | None = None,
        **kwargs: Unpack[RTCActionSelectKwargs],
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks, state, lang_tokens, lang_masks = self._prepare_model_inputs(batch)
        actions = self.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            observation_delay_chunks=batch.get("observation_delay_chunks"),
            observation_delay_tokens=batch.get("observation_delay_tokens"),
            tactile_batch=batch,
            **kwargs,
        )

        return self._postprocess_action_chunk(actions)

    def _prepare_model_inputs(self, batch: dict[str, Tensor]):
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        return images, img_masks, state, lang_tokens, lang_masks

    def _postprocess_action_chunk(self, actions: Tensor) -> Tensor:
        """Crop model padding and restore the checkpoint's action convention."""

        # Unpad actions
        action_feature = self.config.action_feature
        if action_feature is None:
            raise ValueError("SmolVLA requires an action output feature.")
        actions = actions[:, :, : action_feature.shape[0]]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    @torch.no_grad()
    def encode_prefix(self, batch: dict[str, Tensor]) -> dict[str, Any]:
        """Encode the VLM prefix for the split Runtime streaming path."""
        self.eval()
        # Tactile history belongs to the FM-side token and must not be
        # advanced by a VLM-prefix refresh.
        batch = self._prepare_batch(batch, update_tactile_history=False)
        images, img_masks, state, lang_tokens, lang_masks = self._prepare_model_inputs(batch)
        return self.model.encode_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            observation_delay_chunks=batch.get("observation_delay_chunks"),
            observation_delay_tokens=batch.get("observation_delay_tokens"),
            tactile_batch=batch,
        )

    @torch.no_grad()
    def predict_action_chunk_from_prefix(
        self,
        batch: dict[str, Tensor],
        prefix_cache: dict[str, Any],
        *,
        noise: Tensor | None = None,
        **kwargs: Unpack[RTCActionSelectKwargs],
    ) -> Tensor:
        """Run only FM denoising against a previously encoded VLM prefix."""
        self.eval()
        batch = self._prepare_batch(batch)
        actions = self.model.sample_actions_from_prefix(
            prefix_cache,
            noise=noise,
            tactile_batch=batch,
            **kwargs,
        )
        return self._postprocess_action_chunk(actions)

    def _prepare_batch(
        self,
        batch: dict[str, Tensor],
        *,
        update_tactile_history: bool = True,
        delay_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        max_delay = max(self.config.observation_delay_max_chunks, self.config.observation_delay_max_tokens)
        if max_delay:
            key = (
                "observation_delay_tokens"
                if self.config.observation_delay_max_tokens
                else "observation_delay_chunks"
            )
            if key not in batch:
                batch_size = batch[OBS_STATE].shape[0]
                if self.training:
                    batch[key] = torch.randint(
                        0,
                        max_delay + 1,
                        (batch_size,),
                        device=batch[OBS_STATE].device,
                        dtype=torch.long,
                    )
                else:
                    batch[key] = torch.zeros(
                        batch_size,
                        device=batch[OBS_STATE].device,
                        dtype=torch.long,
                    )
            if self.training and delay_mask is not None:
                # Match the reference streaming recipe: constant-time samples
                # always use the current observation; delayed observations are
                # sampled only for chunk-wise/token-wise streaming samples.
                sampled_delay = batch[key].to(device=batch[OBS_STATE].device, dtype=torch.long)
                if sampled_delay.ndim > 1:
                    sampled_delay = sampled_delay[:, -1]
                batch[key] = torch.where(delay_mask, sampled_delay, torch.zeros_like(sampled_delay))

            # When the dataset exposes an observation history, select the
            # historical frame represented by the sampled delay. For a
            # one-step batch this is a no-op, so deployment clients can keep
            # sending the ordinary current-frame contract.
            delay = batch[key].to(device=batch[OBS_STATE].device, dtype=torch.long)
            if delay.ndim > 1:
                delay = delay[:, -1]
            delay_step = (
                self.config.streaming_chunk_size
                if self.config.observation_delay_max_chunks
                else 1
            )
            delay = delay * delay_step
            state = batch[OBS_STATE]
            if state.ndim == 3:
                indices = (state.shape[1] - 1 - delay).clamp(0, state.shape[1] - 1)
                batch[OBS_STATE] = state[torch.arange(state.shape[0], device=state.device), indices]
            for image_key in getattr(self.config, "image_features", {}):
                image = batch.get(image_key)
                if image is not None and image.ndim == 5:
                    indices = (image.shape[1] - 1 - delay).clamp(0, image.shape[1] - 1)
                    batch[image_key] = image[torch.arange(image.shape[0], device=image.device), indices]

        if (
            update_tactile_history
            and self.config.use_tactile
            and self.config.use_tactile_token_history
            and not self.training
        ):
            for side in ("left", "right"):
                current_key = next(
                    (
                        key
                        for key in batch
                        if "tactile" in key.lower()
                        and side in key.lower()
                        and "history" not in key.lower()
                    ),
                    None,
                )
                if current_key is None:
                    continue
                value = batch[current_key]
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value)
                history = self._tactile_history.setdefault(
                    current_key,
                    deque(maxlen=self.config.tactile_token_history_length),
                )
                history.append(value.detach())
                while len(history) < self.config.tactile_token_history_length:
                    history.appendleft(history[0])
                batch[f"{current_key}_history"] = torch.stack(tuple(history), dim=1)

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        *,
        noise: Tensor | None = None,
        **kwargs: Unpack[RTCActionSelectKwargs],
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise=noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
        *,
        noise: Tensor | None = None,
        **kwargs: Unpack[RTCActionSelectKwargs],
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise=noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float]]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        streaming_regime = None
        delay_mask = None
        if self.config.streaming:
            total = self.config.streaming_constant_weight + self.config.streaming_chunk_wise_weight
            draw = torch.rand(batch[OBS_STATE].shape[0], device=batch[OBS_STATE].device) * total
            streaming_regime = (draw >= self.config.streaming_constant_weight).long()
            delay_mask = streaming_regime != 0
        # Apply the same delay sampling/history selection used by the regular
        # policy path before constructing model inputs. Without this call,
        # observation_delay_max_* only existed in the inference helper and had
        # no effect on training batches.
        batch = self._prepare_batch(
            batch, update_tactile_history=False, delay_mask=delay_mask
        )
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")
        loss_dict: dict[str, float] = {}
        losses = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            observation_delay_chunks=batch.get("observation_delay_chunks"),
            observation_delay_tokens=batch.get("observation_delay_tokens"),
            tactile_batch=batch,
            streaming_regime=streaming_regime,
        )
        action_feature = self.config.action_feature
        if action_feature is None:
            raise ValueError("SmolVLA requires an action output feature.")
        losses = losses[:, :, : action_feature.shape[0]]
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        # Remove padding
        losses = losses[:, :, : self.config.action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over valid (time, action) entries
            if actions_is_pad is None:
                per_sample_loss = losses.mean(dim=(1, 2))
            else:
                num_valid = ((~actions_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
                per_sample_loss = losses.sum(dim=(1, 2)) / num_valid
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss over valid (time, action) entries
            if actions_is_pad is None:
                loss = losses.mean()
            else:
                num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
                loss = losses.sum() / num_valid
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def validate_chunk_input_features(self, features: tuple[FeatureSpec, ...]) -> None:
        """Preserve prepare_images' camera subset, resizing and padding semantics.

        The deployment explicitly selects physical cameras. Missing visual slots
        remain missing; prepare_images applies the checkpoint's empty-camera limit
        and masks exactly as in local inference. Never synthesize unmasked cameras.
        """
        expected = self.config.input_features or {}
        supplied = {feature.name: feature for feature in features}
        required = {name for name, feature in expected.items() if feature.type != FeatureType.VISUAL}
        if not required <= supplied.keys() or not supplied.keys() <= expected.keys():
            raise ValueError("SmolVLA requires all nonvisual inputs and only known camera features.")
        if not any(expected[name].type == FeatureType.VISUAL for name in supplied):
            raise ValueError("SmolVLA requires at least one camera.")
        for name, feature in supplied.items():
            policy_feature = expected[name]
            if policy_feature.type == FeatureType.VISUAL:
                shape = policy_feature.shape
                if len(shape) != 3 or shape[0] != 3 or feature.kind != "rgb":
                    raise ValueError(f"SmolVLA requires RGB camera inputs: {name}.")
                if self.config.resize_imgs_with_padding is None and feature.shape != (
                    shape[1],
                    shape[2],
                    shape[0],
                ):
                    raise ValueError(f"RGB feature shape differs without policy resizing: {name}.")
            elif feature.kind != "tensor" or feature.shape != tuple(policy_feature.shape):
                raise ValueError(f"Tensor feature shape differs from checkpoint: {name}.")

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                "All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if img.dtype == torch.uint8:
                img = img.to(torch.float32) / 255.0
            if self.config.resize_imgs_with_padding is not None:
                # SmolVLA stores the target as (width, height); the shared helper expects (height, width).
                img = resize_with_pad(
                    img,
                    self.config.resize_imgs_with_padding[1],
                    self.config.resize_imgs_with_padding[0],
                    pad_value=0,
                )

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state to the configured dynamic state projection dimension."""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        if state.shape[-1] > self.config.state_dim:
            raise ValueError(
                f"SmolVLA received state dimension {state.shape[-1]}, but state_dim={self.config.state_dim}. "
                "Set --policy.state_dim to the dataset state dimension."
            )
        state = pad_vector(state, self.config.state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action to the configured dynamic action projection dimension."""
        if batch[ACTION].shape[-1] > self.config.action_dim:
            raise ValueError(
                f"SmolVLA received action dimension {batch[ACTION].shape[-1]}, "
                f"but action_dim={self.config.action_dim}. "
                "Set --policy.action_dim to the dataset action dimension."
            )
        actions = pad_vector(batch[ACTION], self.config.action_dim)
        return actions

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """Return default PEFT target modules for SmolVLA fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out|"
            "observation_delay_embedding|observation_delay_adarms|tactile_encoder|tactile_adarms|"
            "tactile_token_proj|tactile_token_tcn_in|tactile_token_tcn_1|"
            "tactile_token_tcn_2|tactile_token_out"
        )
        target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    def _validate_peft_config(self, peft_config) -> None:
        """Validate PEFT configuration for SmolVLA."""
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging

            logging.warning(
                "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        # The pretrained SmolVLM loader uses bfloat16 weights by default, while
        # the action expert and auxiliary projections are initialized in float32.
        # FSDP requires uniform original dtypes within each wrapped unit; keep
        # the default policy dtype uniform when no explicit dtype was requested.
        if self.config.dtype in (None, torch.float32):
            self.vlm_with_expert.float()
        self.state_proj = nn.Linear(
            self.config.state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        self._delay_max = max(
            self.config.observation_delay_max_chunks,
            self.config.observation_delay_max_tokens,
        )
        self.observation_delay_embedding = (
            nn.Embedding(self._delay_max + 1, self.vlm_with_expert.expert_hidden_size)
            if self.config.use_observation_delay_embedding and self._delay_max > 0
            else None
        )
        self.observation_delay_adarms = (
            nn.Sequential(
                nn.Linear(1, self.vlm_with_expert.expert_hidden_size),
                nn.SiLU(),
                nn.Linear(self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size),
            )
            if self.config.use_observation_delay_adarms and self._delay_max > 0
            else None
        )
        if self.config.use_tactile:
            # LazyLinear keeps the policy compatible with both the UniVTAC
            # marker layout (2 x 63 x 2 per hand) and flattened tactile arrays
            # from other datasets without hard-coding a sensor width.
            tactile_hidden = self.vlm_with_expert.config.text_config.hidden_size
            self.tactile_encoder = nn.Sequential(
                nn.LazyLinear(tactile_hidden),
                nn.SiLU(),
                nn.Linear(tactile_hidden, tactile_hidden),
            )
            # The Pi05 tactile token belongs to the FM/action-expert stream.
            # Keep the legacy encoder width so existing checkpoints can still
            # provide the raw marker projection, then adapt it to the expert
            # width before constructing the dynamic FM suffix token.
            self.tactile_token_proj = nn.Linear(tactile_hidden, self.vlm_with_expert.expert_hidden_size)
            if self.config.use_tactile_token_history:
                self.tactile_tcn_width = self.config.tactile_tcn_width
                self.tactile_token_tcn_in = nn.Linear(
                    self.vlm_with_expert.expert_hidden_size, self.tactile_tcn_width
                )
                self.tactile_token_tcn_1 = nn.Linear(
                    3 * self.tactile_tcn_width, self.tactile_tcn_width
                )
                self.tactile_token_tcn_2 = nn.Linear(
                    3 * self.tactile_tcn_width, self.tactile_tcn_width
                )
                self.tactile_token_out = nn.Linear(
                    self.tactile_tcn_width, self.vlm_with_expert.expert_hidden_size
                )
            else:
                self.tactile_tcn_width = None
                self.tactile_token_tcn_in = None
                self.tactile_token_tcn_1 = None
                self.tactile_token_tcn_2 = None
                self.tactile_token_out = None
            self.tactile_adarms = (
                nn.Sequential(
                    nn.Linear(tactile_hidden, self.vlm_with_expert.expert_hidden_size),
                    nn.SiLU(),
                    nn.Linear(self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size),
                )
                if self.config.use_tactile_adarms
                else None
            )
        else:
            self.tactile_encoder = None
            self.tactile_adarms = None

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor
        self._stream_head_state: dict[str, Tensor] | None = None

        # Compile model if requested
        if config.compile_model:
            self.sample_actions = torch.compile(  # type: ignore[method-assign]
                self.sample_actions,
                mode=config.compile_mode,
            )
            self.forward = torch.compile(self.forward, mode=config.compile_mode)  # type: ignore[method-assign]

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        return sample_noise(shape, device)

    def sample_time(self, bsize, device):
        return sample_time_beta(bsize, device, alpha=1.5, beta=1.0, scale=0.999, offset=0.001)

    def _delay_tensor(self, delay_chunks, delay_tokens, *, batch_size, device):
        if self._delay_max <= 0:
            return torch.zeros(batch_size, device=device, dtype=torch.long)
        delay = delay_tokens if self.config.observation_delay_max_tokens else delay_chunks
        if delay is None:
            delay = torch.zeros(batch_size, device=device, dtype=torch.long)
        if delay.ndim > 1:
            delay = delay[:, -1]
        return delay.to(device=device, dtype=torch.long).clamp(0, self._delay_max)

    def _find_tactile_key(self, batch, side: str, *, history: bool = False):
        names = list(self.config.tactile_feature_names)
        suffix = "_history" if history else ""
        candidates = []
        for name in names:
            if side in name.lower():
                candidates.extend([f"{name}{suffix}", f"{name.replace('.', '_')}{suffix}"])
        candidates.extend(
            [
                f"observation.tactile.{side}_marker{suffix}",
                f"observation.tactile_{side}_marker{suffix}",
                f"tactile_{side}_marker{suffix}",
                f"tactile.{side}_marker{suffix}",
            ]
        )
        for candidate in candidates:
            if candidate in batch:
                return candidate
        for key in batch:
            lowered = key.lower()
            if "tactile" in lowered and side in lowered and (not history or "history" in lowered):
                return key
        return None

    @staticmethod
    def _causal_conv_inputs(x: Tensor, *, dilation: int) -> Tensor:
        """Build the three causal taps used by the Pi05 tactile TCN."""
        length = x.shape[1]
        taps = []
        for lag in (2 * dilation, dilation, 0):
            if lag == 0:
                taps.append(x)
            elif lag >= length:
                taps.append(x[:, :0].new_zeros(x.shape[0], length, x.shape[2]))
            else:
                taps.append(F.pad(x[:, :-lag], (0, 0, lag, 0)))
        return torch.cat(taps, dim=-1)

    def _tactile_frame_values(self, batch) -> tuple[Tensor, Tensor]:
        """Return left/right marker frames as ``[B, H, ...]`` tensors."""
        values = []
        for side in ("left", "right"):
            key = self._find_tactile_key(batch, side)
            if key is None:
                raise ValueError(
                    "use_tactile=True requires left and right tactile marker features; "
                    f"available keys: {sorted(batch)}"
                )
            value = batch[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            value = value.to(dtype=torch.float32)
            original_value = value
            # Runtime inputs are one current frame. Training inputs may carry
            # an observation axis; the tactile history is handled separately.
            if value.ndim >= 5:
                value = value[:, -1:]
            elif value.ndim == 4:
                value = value[:, None]
            elif value.ndim == 3:
                value = value[None, None]
            else:
                raise ValueError(f"Unsupported tactile marker shape for {key}: {tuple(value.shape)}")
            history_key = self._find_tactile_key(batch, side, history=True)
            if self.config.use_tactile_token_history and history_key is not None:
                history = batch[history_key]
                if not torch.is_tensor(history):
                    history = torch.as_tensor(history)
                value = history.to(dtype=torch.float32)
                if value.ndim < 5:
                    raise ValueError("Tactile history must have shape (B, H, ...)")
                expected = self.config.tactile_token_history_length
                if value.shape[1] != expected:
                    raise ValueError(f"Expected tactile history length {expected}, got {value.shape[1]}")
            elif self.config.use_tactile_token_history and original_value.ndim >= 5:
                value = original_value[:, -self.config.tactile_token_history_length :]
            if self.config.use_tactile_token_history and history_key is None:
                expected = self.config.tactile_token_history_length
                if value.shape[1] == 1 and expected > 1:
                    value = value.expand(-1, expected, *value.shape[2:])
                elif value.shape[1] != expected:
                    value = value[:, -expected:]
            values.append(value)
        return values[0], values[1]

    def _encode_tactile_frames(self, batch) -> Tensor:
        """Encode each tactile frame independently before temporal fusion."""
        if not self.config.use_tactile or not (
            self.config.use_tactile_token or self.config.use_tactile_adarms
        ):
            return None
        if self.tactile_encoder is None:
            raise RuntimeError("Tactile encoder was not initialized")
        left, right = self._tactile_frame_values(batch)
        batch_size, history = left.shape[:2]
        frame_tactile = torch.cat(
            [left.flatten(start_dim=2), right.flatten(start_dim=2)], dim=-1
        )
        flattened = frame_tactile.reshape(batch_size * history, -1)
        first_layer = self.tactile_encoder[0]
        expected_width = getattr(first_layer, "in_features", None)
        if expected_width not in (None, 0, flattened.shape[-1]):
            raise ValueError(
                "Tactile encoder input width differs from the checkpoint: "
                f"expected {expected_width}, frame width {flattened.shape[-1]}"
            )
        encoded = self.tactile_encoder(flattened)
        return encoded.reshape(batch_size, history, -1)

    def _encode_tactile(self, batch):
        """Encode the current tactile observation into an FM-side token.

        This method is intentionally called from the FM path. The VLM prefix
        cache never stores this token, so every fresh inference observation can
        replace it without recomputing the visual/language prefix.
        """
        frames = self._encode_tactile_frames(batch)
        if frames is None:
            return None
        token_proj = getattr(self, "tactile_token_proj", None)
        expert_frames = token_proj(frames) if token_proj is not None else frames
        tcn_in = getattr(self, "tactile_token_tcn_in", None)
        if self.config.use_tactile_token_history and tcn_in is not None:
            x = F.gelu(self.tactile_token_tcn_in(expert_frames))
            x = F.gelu(x + self.tactile_token_tcn_1(self._causal_conv_inputs(x, dilation=1)))
            x = F.gelu(x + self.tactile_token_tcn_2(self._causal_conv_inputs(x, dilation=2)))
            return self.tactile_token_out(x[:, -1, :])
        return expert_frames[:, -1, :]

    def _encode_tactile_adarms(self, batch):
        """Encode the current frame for optional AdaRMS conditioning."""
        frames = self._encode_tactile_frames(batch)
        if frames is None:
            return None
        return frames[:, -1, :]

    def _action_att_masks(self, batch_size: int, device, dtype):
        """Return action block boundaries for the requested streaming mode."""
        horizon = self.config.chunk_size
        if not self.config.streaming:
            boundaries = torch.ones(batch_size, horizon, device=device, dtype=dtype)
            return boundaries
        mode = self.config.streaming_attention_mode
        if mode in ("bidirectional", "mask"):
            boundaries = torch.zeros(batch_size, horizon, device=device, dtype=dtype)
            boundaries[:, 0] = 1
            return boundaries
        if mode == "causal":
            boundaries = torch.zeros(batch_size, horizon, device=device, dtype=dtype)
            boundaries[:, :: self.config.streaming_chunk_size] = 1
            return boundaries
        # random_gate uses a soft additive gate during training. Keep the
        # structural mask bidirectional so inference can use the same layout
        # without sampling a gate.
        boundaries = torch.zeros(batch_size, horizon, device=device, dtype=dtype)
        boundaries[:, 0] = 1
        return boundaries

    def _random_attention_log_gates(self, batch_size: int, device, dtype) -> Tensor:
        """Sample Master_Research-style continuous inter-chunk attention gates.

        The sampled gate is uniform on [epsilon, e], then converted to an
        additive attention bias with ``log(gate)``. Values above one therefore
        strengthen cross-chunk attention, while values below one weaken it.
        """
        chunk_count = self.config.chunk_size // self.config.streaming_chunk_size
        epsilon = torch.finfo(torch.float32).tiny
        gates = torch.empty(batch_size, chunk_count, device=device, dtype=torch.float32).uniform_(epsilon, math.e)
        return gates.log().to(dtype=dtype)

    def _make_action_attention_bias(
        self,
        log_gates: Tensor | None,
        *,
        query_length: int,
        key_length: int,
        action_query_start: int,
        action_key_start: int,
        action_count: int,
    ) -> Tensor | None:
        """Build the additive bias used by random_gate during training."""
        if log_gates is None or self.config.use_tactile_token:
            return None
        chunk_size = self.config.streaming_chunk_size
        chunk_ids = torch.arange(action_count, device=log_gates.device) // chunk_size
        same_chunk = chunk_ids[:, None] == chunk_ids[None, :]
        query_gate = log_gates[:, chunk_ids]
        action_bias = torch.where(same_chunk[None], torch.zeros_like(query_gate[:, :, None]), query_gate[:, :, None])
        bias = torch.zeros(
            log_gates.shape[0], query_length, key_length, device=log_gates.device, dtype=log_gates.dtype
        )
        bias[
            :,
            action_query_start : action_query_start + action_count,
            action_key_start : action_key_start + action_count,
        ] = action_bias
        return bias

    @staticmethod
    def _apply_tactile_token_mask(
        attention_mask: Tensor,
        *,
        prefix_len: int,
        action_count: int,
        query_offset: int = 0,
    ) -> Tensor:
        """Apply Pi05's isolated ``T, CA0, A0`` visibility rule."""
        if action_count < 1:
            raise ValueError("tactile token masking requires at least one action")
        tactile_index = prefix_len
        copy_action_index = prefix_len + 1
        action0_index = prefix_len + 2
        if attention_mask.shape[-1] < action0_index + action_count:
            raise ValueError("attention mask is shorter than the tactile-token suffix")
        tactile_query_index = tactile_index - query_offset
        copy_query_index = copy_action_index - query_offset
        action0_query_index = action0_index - query_offset
        if not 0 <= tactile_query_index < attention_mask.shape[1] or not 0 <= action0_query_index < attention_mask.shape[1]:
            raise ValueError("attention mask query rows do not cover the tactile-token suffix")

        masked = attention_mask.clone()
        # No query can consume the special keys by default.
        masked[:, :, tactile_index] = False
        masked[:, :, copy_action_index] = False
        masked[:, :, action0_index] = False

        tactile_prefix = masked[:, tactile_query_index, :prefix_len].clone()
        copy_prefix = masked[:, copy_query_index, :prefix_len].clone()
        action0_prefix = masked[:, action0_query_index, :prefix_len].clone()

        # T reads the complete VLM prefix and itself, never an action.
        masked[:, tactile_query_index, :] = False
        masked[:, tactile_query_index, :prefix_len] = tactile_prefix
        masked[:, tactile_query_index, tactile_index] = True

        # CA0 reads prefix + T + itself.
        masked[:, copy_query_index, :] = False
        masked[:, copy_query_index, :prefix_len] = copy_prefix
        masked[:, copy_query_index, tactile_index] = True
        masked[:, copy_query_index, copy_action_index] = True

        # A0 reads prefix + its own key, but neither T nor CA0.
        masked[:, action0_query_index, :] = False
        masked[:, action0_query_index, :prefix_len] = action0_prefix
        masked[:, action0_query_index, action0_index] = True
        return masked

    @staticmethod
    def _apply_head_guidance(
        velocity: Tensor,
        *,
        inference_mode: str,
        context_cfg_lambda: float,
        chunk_size: int,
        streaming_chunk_size: int | None = None,
    ) -> Tensor:
        """Update only chunk heads for the head-streaming inference modes.

        ``chunk_size`` is the full action horizon. ``streaming_chunk_size`` is
        the temporal block size, so heads are at positions 0, C, 2C, ... .
        """
        if inference_mode not in ("stream-head", "stream-head-context-cfg", "stream-head-mean"):
            return velocity
        block = int(streaming_chunk_size or chunk_size)
        if block <= 0:
            raise ValueError("streaming_chunk_size must be positive")
        head_indices = torch.arange(0, velocity.shape[1], block, device=velocity.device)
        guided = torch.zeros_like(velocity)
        head_velocity = velocity[:, head_indices]
        if inference_mode == "stream-head":
            guided[:, head_indices] = head_velocity
            return guided
        if inference_mode == "stream-head-mean":
            # Mean along the same relative token time across chunks:
            # v_0, v_C, v_2C, ... . This is deliberately not a mean over
            # different token positions within one chunk.
            mean_head = head_velocity.mean(dim=1, keepdim=True)
            guided[:, head_indices] = mean_head.expand_as(head_velocity)
            return guided

        # Context-CFG uses the later token fields belonging to each same
        # chunk. Only chunk heads are updated; non-head tokens remain fixed.
        for head in head_indices.tolist():
            end = min(head + block, velocity.shape[1])
            context = velocity[:, head + 1 : end]
            current = velocity[:, head : head + 1]
            if context.shape[1] == 0:
                guided[:, head : head + 1] = current
                continue
            guided[:, head : head + 1] = current + context_cfg_lambda * (
                context.sum(dim=1, keepdim=True) - context.shape[1] * current
            )
        return guided

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: torch.Tensor = None,
        observation_delay_chunks=None,
        observation_delay_tokens=None,
        tactile_batch=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        delay = self._delay_tensor(
            observation_delay_chunks,
            observation_delay_tokens,
            batch_size=bsize,
            device=device,
        )

        # Tactile tokens are FM-side conditions. They are deliberately kept
        # out of the reusable VLM prefix so a new observation can refresh the
        # token without invalidating the visual/language cache.
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(
        self,
        noisy_actions,
        timestep,
        observation_delay_chunks=None,
        observation_delay_tokens=None,
        tactile_embedding=None,
        tactile_condition=None,
    ):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        if time_emb.ndim == 2:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
        elif time_emb.shape[:2] != action_emb.shape[:2]:
            time_emb = time_emb[:, : action_emb.shape[1], :]
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        delay = self._delay_tensor(
            observation_delay_chunks,
            observation_delay_tokens,
            batch_size=bsize,
            device=device,
        )

        # Delay is a dynamic FM condition. Keep it out of the reusable VLM
        # prefix so a lagging prefix can be paired with the current cache-age
        # value on every action-expert invocation.
        if self.observation_delay_embedding is not None:
            action_time_emb = action_time_emb + self.observation_delay_embedding(delay)[:, None, :].to(
                action_time_emb.dtype
            )

        if self.observation_delay_adarms is not None:
            delay_condition = self.observation_delay_adarms(
                (delay.to(dtype=action_time_emb.dtype) / max(self._delay_max, 1)).unsqueeze(-1)
            )
            action_time_emb = action_time_emb + delay_condition[:, None, :].to(action_time_emb.dtype)
        if tactile_condition is not None and self.tactile_adarms is not None:
            action_time_emb = action_time_emb + self.tactile_adarms(tactile_condition)[:, None, :].to(
                action_time_emb.dtype
            )

        # The Pi05 tactile-token suffix is ``T, CA0, A0, A1, ...``. CA0 is a
        # copy of A0 with the same noisy input and timestep condition, but its
        # attention row is isolated so only CA0 consumes tactile context.
        use_tactile_token = tactile_embedding is not None and self.config.use_tactile_token
        action_tokens = action_time_emb
        action_att_masks = self._action_att_masks(bsize, device, torch.bool)
        if use_tactile_token:
            tactile_token = tactile_embedding[:, None, :].to(dtype=action_time_emb.dtype)
            conditioned_copy = action_time_emb[:, :1]
            action_tokens = torch.cat([conditioned_copy, action_time_emb], dim=1)
            embs.extend([tactile_token, action_tokens])
            pad_masks.extend(
                [
                    torch.ones(bsize, 1, dtype=torch.bool, device=device),
                    torch.ones(bsize, action_tokens.shape[1], dtype=torch.bool, device=device),
                ]
            )
            # T shares the prefix block; CA0 starts a private block and the
            # real actions retain the configured streaming block boundaries.
            att_masks += [0, 1] + action_att_masks[0].tolist()
        else:
            embs.append(action_tokens)
            pad_masks.append(torch.ones(bsize, action_tokens.shape[1], dtype=torch.bool, device=device))
            att_masks += action_att_masks[0].tolist()
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=device)[None, :].expand(bsize, -1)
        return embs, pad_masks, att_masks

    def _sample_training_time(
        self, batch_size: int, horizon: int, device, streaming_regime: Tensor | None = None
    ) -> Tensor:
        if not self.config.streaming:
            return self.sample_time(batch_size, device)
        total = self.config.streaming_constant_weight + self.config.streaming_chunk_wise_weight
        draw = torch.rand(batch_size, device=device) * total
        constant = self.sample_time(batch_size, device)[:, None].expand(batch_size, horizon)
        num_chunks = horizon // self.config.streaming_chunk_size
        # Match Pi05's diffusion-forcing schedule: chunk j samples inside its
        # own equal-width interval, then shares that timestep across the chunk.
        chunk_samples = self.sample_time(batch_size * num_chunks, device).reshape(batch_size, num_chunks)
        starts = torch.arange(num_chunks, device=device, dtype=chunk_samples.dtype)[None, :] / num_chunks
        chunk_times = starts + chunk_samples / num_chunks
        chunk_times = chunk_times.repeat_interleave(self.config.streaming_chunk_size, dim=1)
        constant_mask = (
            streaming_regime == 0
            if streaming_regime is not None
            else draw < self.config.streaming_constant_weight
        )
        return torch.where(
            constant_mask[:, None],
            constant,
            chunk_times,
        )

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        *,
        observation_delay_chunks=None,
        observation_delay_tokens=None,
        tactile_batch=None,
        streaming_regime: Tensor | None = None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self._sample_training_time(
                actions.shape[0], actions.shape[1], actions.device, streaming_regime=streaming_regime
            )

        if time.ndim == 1:
            expanded_time = time[:, None, None]
            model_time = time
        else:
            expanded_time = time[:, :, None]
            model_time = time
        x_t = expanded_time * noise + (1 - expanded_time) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
            observation_delay_chunks=observation_delay_chunks,
            observation_delay_tokens=observation_delay_tokens,
            tactile_batch=tactile_batch,
        )
        tactile_embedding = self._encode_tactile(tactile_batch) if tactile_batch is not None else None
        tactile_condition = (
            self._encode_tactile_adarms(tactile_batch)
            if tactile_batch is not None and self.tactile_adarms is not None
            else None
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            x_t,
            model_time,
            observation_delay_chunks=observation_delay_chunks,
            observation_delay_tokens=observation_delay_tokens,
            tactile_embedding=tactile_embedding,
            tactile_condition=tactile_condition,
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        if tactile_embedding is not None and self.config.use_tactile_token:
            att_2d_masks = self._apply_tactile_token_mask(
                att_2d_masks,
                prefix_len=prefix_pad_masks.shape[1],
                action_count=self.config.chunk_size,
            )
        attention_bias = None
        if self.training and self.config.streaming and self.config.streaming_attention_mode == "random_gate":
            attention_log_gates = self._random_attention_log_gates(
                actions.shape[0], actions.device, torch.float32
            )
            action_start = prefix_pad_masks.shape[1] + suffix_pad_masks.shape[1] - self.config.chunk_size
            attention_bias = self._make_action_attention_bias(
                attention_log_gates,
                query_length=att_2d_masks.shape[1],
                key_length=att_2d_masks.shape[2],
                action_query_start=action_start,
                action_key_start=action_start,
                action_count=self.config.chunk_size,
            )
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        if tactile_embedding is not None and self.config.use_tactile_token:
            copy_index = prefix_pad_masks.shape[1] + 1
            action0_index = prefix_pad_masks.shape[1] + 2
            position_ids[:, action0_index] = position_ids[:, copy_index]
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            attention_bias=attention_bias,
        )
        copy_out = suffix_out[:, -(self.config.chunk_size + 1)] if tactile_embedding is not None else None
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        if copy_out is not None:
            copy_v_t = self.action_out_proj(copy_out.to(dtype=torch.float32))
            copy_loss = F.mse_loss(copy_v_t, u_t[:, 0], reduction="none")
            losses[:, 0] = 0.5 * (losses[:, 0] + copy_loss)
        return losses

    def encode_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        *,
        observation_delay_chunks=None,
        observation_delay_tokens=None,
        tactile_batch=None,
    ) -> dict[str, Any]:
        """Run the VLM/prefix stage and return a reusable KV cache."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
            observation_delay_chunks=observation_delay_chunks,
            observation_delay_tokens=observation_delay_tokens,
            tactile_batch=tactile_batch,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
        )
        return {
            "prefix_pad_masks": prefix_pad_masks,
            "past_key_values": past_key_values,
            "observation_delay_chunks": observation_delay_chunks,
            "observation_delay_tokens": observation_delay_tokens,
        }

    def sample_actions_from_prefix(
        self,
        prefix_cache: dict[str, Any],
        noise=None,
        tactile_batch=None,
        **kwargs: Unpack[RTCActionSelectKwargs],
    ) -> Tensor:
        """Run only the FM/action denoising stage against a prefix cache."""
        prefix_pad_masks = prefix_cache["prefix_pad_masks"]
        past_key_values = prefix_cache["past_key_values"]
        bsize = prefix_pad_masks.shape[0]
        device = prefix_pad_masks.device
        if noise is None:
            noise = self.sample_noise((bsize, self.config.chunk_size, self.config.action_dim), device)
        else:
            noise = noise.to(device=device, dtype=torch.float32)

        observation_delay_chunks = kwargs.get("observation_delay_chunks", prefix_cache.get("observation_delay_chunks"))
        observation_delay_tokens = kwargs.get("observation_delay_tokens", prefix_cache.get("observation_delay_tokens"))
        inference_mode = kwargs.get("inference_mode", "stream")
        context_cfg_lambda = float(kwargs.get("context_cfg_lambda", 0.1))
        if inference_mode not in (
            "diffusion",
            "stream",
            "stream-head",
            "stream-head-context-cfg",
            "stream-head-mean",
        ):
            raise ValueError(f"Unsupported SmolVLA streaming inference mode: {inference_mode}")
        if context_cfg_lambda < 0 or not math.isfinite(context_cfg_lambda):
            raise ValueError("context_cfg_lambda must be finite and non-negative")
        if inference_mode == "stream":
            return self.stream_step_from_prefix(
                prefix_cache,
                tactile_batch=tactile_batch,
                noise=noise,
                observation_delay_chunks=observation_delay_chunks,
                observation_delay_tokens=observation_delay_tokens,
            )
        if inference_mode in ("stream-head", "stream-head-context-cfg", "stream-head-mean"):
            return self.stream_head_step_from_prefix(
                prefix_cache,
                tactile_batch=tactile_batch,
                noise=noise,
                inference_mode=inference_mode,
                context_cfg_lambda=context_cfg_lambda,
                observation_delay_chunks=observation_delay_chunks,
                observation_delay_tokens=observation_delay_tokens,
            )
        # Recompute tactile conditions on the FM side for every fresh
        # observation. They are intentionally absent from ``prefix_cache``.
        tactile_embedding = self._encode_tactile(tactile_batch) if tactile_batch is not None else None
        tactile_condition = (
            self._encode_tactile_adarms(tactile_batch)
            if tactile_batch is not None and self.tactile_adarms is not None
            else None
        )
        return euler_integrate(
            lambda input_x_t, current_timestep: self.denoise_step(
                x_t=input_x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=current_timestep,
                observation_delay_chunks=observation_delay_chunks,
                observation_delay_tokens=observation_delay_tokens,
                tactile_embedding=tactile_embedding,
                tactile_condition=tactile_condition,
                inference_mode=inference_mode,
                context_cfg_lambda=context_cfg_lambda,
            ),
            noise,
            self.config.num_steps,
            rtc_processor=self.rtc_processor,
            rtc_enabled=self._rtc_enabled(),
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
        )

    def reset_stream_head_state(self) -> None:
        """Clear token-aligned sliding-window state for a new runtime session."""
        self._stream_head_state = None
        self._stream_state = None

    def stream_step_from_prefix(
        self,
        prefix_cache: dict[str, Any],
        *,
        tactile_batch=None,
        noise: Tensor | None = None,
        observation_delay_chunks=None,
        observation_delay_tokens=None,
    ) -> Tensor:
        """Advance the diffusion-forcing window by one execution chunk.

        The first call performs a normal full-horizon FM sample, then assigns
        the deterministic chunk-wise timestep schedule to the future window.
        Each later call cleans and returns one whole streaming chunk, shifts the
        window by that chunk, injects a fresh noisy chunk, and transports the
        remaining chunks. The internal window has ``horizon + block`` tokens:
        the leading block is ready for execution, while the FM model receives
        the following full horizon without that already-clean block.
        """
        prefix_pad_masks = prefix_cache["prefix_pad_masks"]
        bsize = prefix_pad_masks.shape[0]
        device = prefix_pad_masks.device
        horizon = self.config.chunk_size
        block = self.config.streaming_chunk_size if self.config.streaming else horizon
        if horizon % block:
            raise ValueError("chunk_size must be divisible by streaming_chunk_size")
        state = self._stream_state
        valid_state = (
            state is not None
            and state["action_window"].shape[:2] == (bsize, horizon + block)
            and state["action_window"].device == device
        )
        tactile_embedding = self._encode_tactile(tactile_batch) if tactile_batch is not None else None
        tactile_condition = (
            self._encode_tactile_adarms(tactile_batch)
            if tactile_batch is not None and self.tactile_adarms is not None
            else None
        )
        if not valid_state:
            if noise is None:
                noise = self.sample_noise((bsize, horizon, self.config.action_dim), device)
            else:
                noise = noise.to(device=device, dtype=torch.float32).clone()
            clean_actions = euler_integrate(
                lambda input_x_t, current_timestep: self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=prefix_cache["past_key_values"],
                    timestep=current_timestep,
                    observation_delay_chunks=observation_delay_chunks,
                    observation_delay_tokens=observation_delay_tokens,
                    tactile_embedding=tactile_embedding,
                    tactile_condition=tactile_condition,
                    inference_mode="stream",
                ),
                noise,
                self.config.num_steps,
            )
            timestep_window = self._streaming_timestep_schedule(device=device, dtype=torch.float32)
            # Keep the first execution chunk clean. The model-facing future
            # window uses the complete H-token schedule; the trailing block is
            # fresh noise that will enter after the next shift.
            if horizon < block:
                raise ValueError("Streaming action horizon must be at least 2")
            future = (
                timestep_window[:-block][None, :, None] * noise[:, :-block]
                + (1.0 - timestep_window[:-block][None, :, None]) * clean_actions[:, block:]
            )
            action_window = torch.cat([clean_actions[:, :block], future, noise[:, -block:]], dim=1)
        else:
            action_window = state["action_window"]
            # Match Master_Research: remove the already-ready chunk before the
            # FM call, so the model always sees exactly H future tokens.
            model_actions = action_window[:, block:]
            timestep_window = self._streaming_timestep_schedule(
                device=device, dtype=action_window.dtype
            )
            raw_velocity = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=prefix_cache["past_key_values"],
                x_t=model_actions,
                timestep=timestep_window[None, :].expand(bsize, -1),
                observation_delay_chunks=observation_delay_chunks,
                observation_delay_tokens=observation_delay_tokens,
                tactile_embedding=tactile_embedding,
                tactile_condition=tactile_condition,
                inference_mode="stream",
                apply_head_guidance=False,
            )
            # Clean the complete outgoing chunk, then transport each future
            # chunk by the timestep difference caused by the block shift.
            outgoing = model_actions[:, :block] - timestep_window[:block][None, :, None] * raw_velocity[:, :block]
            future = model_actions[:, block:] + (
                timestep_window[:-block][None, :, None] - timestep_window[block:][None, :, None]
            ) * raw_velocity[:, block:]
            fresh_noise = self.sample_noise((bsize, block, self.config.action_dim), device)
            action_window = torch.cat([outgoing, future, fresh_noise], dim=1)
        self._stream_state = {
            "action_window": action_window.detach(),
        }
        # Keep the policy contract at the configured prediction horizon. The
        # Runtime consumes only the first execution chunk and queues its
        # remaining actions; future positions stay internal diffusion state.
        return action_window[:, :horizon]

    def _streaming_timestep_schedule(self, *, device, dtype) -> Tensor:
        """Return the deterministic equal-width token-wise stream schedule."""
        block = self.config.streaming_chunk_size if self.config.streaming else self.config.chunk_size
        num_chunks, remainder = divmod(int(self.config.chunk_size), int(block))
        if remainder:
            raise ValueError("chunk_size must be divisible by streaming_chunk_size for head streaming")
        chunk_time = (torch.arange(num_chunks, device=device, dtype=dtype) + 1.0) / num_chunks
        return torch.repeat_interleave(chunk_time, block)

    def stream_head_step_from_prefix(
        self,
        prefix_cache: dict[str, Any],
        *,
        tactile_batch=None,
        noise: Tensor | None = None,
        inference_mode: str = "stream-head",
        context_cfg_lambda: float = 0.1,
        observation_delay_chunks=None,
        observation_delay_tokens=None,
    ) -> Tensor:
        """Advance each chunk head while carrying field history with tokens.

        The action window is indexed by position (``v0``, ``v1``, ...), while
        ``history`` is attached to the action token currently occupying that
        position. On every control step the window shifts left, so a token's
        saved fields move with it. When a token reaches a chunk head, its
        current field is combined with its own saved fields from the preceding
        positions in that chunk.
        """
        if inference_mode not in ("stream-head", "stream-head-context-cfg", "stream-head-mean"):
            raise ValueError(f"Unsupported stream-head mode: {inference_mode}")
        if context_cfg_lambda < 0 or not math.isfinite(context_cfg_lambda):
            raise ValueError("context_cfg_lambda must be finite and non-negative")

        prefix_pad_masks = prefix_cache["prefix_pad_masks"]
        bsize = prefix_pad_masks.shape[0]
        device = prefix_pad_masks.device
        horizon = self.config.chunk_size
        block = self.config.streaming_chunk_size if self.config.streaming else horizon
        history_length = max(block - 1, 0)
        state = self._stream_head_state
        valid_state = (
            state is not None
            and state["action_window"].shape[:2] == (bsize, horizon)
            and state["action_window"].device == device
        )
        if not valid_state:
            if noise is None:
                noise = self.sample_noise((bsize, horizon, self.config.action_dim), device)
            else:
                noise = noise.to(device=device, dtype=torch.float32).clone()
            tactile_embedding = self._encode_tactile(tactile_batch) if tactile_batch is not None else None
            tactile_condition = (
                self._encode_tactile_adarms(tactile_batch)
                if tactile_batch is not None and self.tactile_adarms is not None
                else None
            )
            clean_actions = euler_integrate(
                lambda input_x_t, current_timestep: self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=prefix_cache["past_key_values"],
                    timestep=current_timestep,
                    observation_delay_chunks=observation_delay_chunks,
                    observation_delay_tokens=observation_delay_tokens,
                    tactile_embedding=tactile_embedding,
                    tactile_condition=tactile_condition,
                    inference_mode="stream",
                    context_cfg_lambda=context_cfg_lambda,
                    apply_head_guidance=False,
                ),
                noise,
                self.config.num_steps,
            )
            timestep_window = self._streaming_timestep_schedule(
                device=device, dtype=torch.float32
            )[None, :].expand(bsize, -1).clone()
            action_window = (
                timestep_window[..., None] * noise
                + (1.0 - timestep_window[..., None]) * clean_actions
            )
            history = torch.zeros(
                (bsize, horizon, history_length, action_window.shape[-1]),
                dtype=action_window.dtype,
                device=device,
            )
            history_valid = torch.zeros((bsize, horizon, history_length), dtype=torch.bool, device=device)
        else:
            action_window = state["action_window"][:, 1:].clone()
            timestep_window = state["timestep_window"][:, 1:].clone()
            history = state["history"][:, 1:].clone()
            history_valid = state["history_valid"][:, 1:].clone()
            fresh_noise = self.sample_noise((bsize, 1, self.config.action_dim), device)
            action_window = torch.cat([action_window, fresh_noise], dim=1)
            timestep_window = torch.cat(
                [timestep_window, torch.ones((bsize, 1), dtype=timestep_window.dtype, device=device)], dim=1
            )
            if history_length:
                history = torch.cat(
                    [
                        history,
                        torch.zeros(
                            (bsize, 1, history_length, action_window.shape[-1]),
                            dtype=history.dtype,
                            device=device,
                        ),
                    ],
                    dim=1,
                )
                history_valid = torch.cat(
                    [
                        history_valid,
                        torch.zeros((bsize, 1, history_length), dtype=torch.bool, device=device),
                    ],
                    dim=1,
                )

        if valid_state:
            tactile_embedding = self._encode_tactile(tactile_batch) if tactile_batch is not None else None
            tactile_condition = (
                self._encode_tactile_adarms(tactile_batch)
                if tactile_batch is not None and self.tactile_adarms is not None
                else None
            )
        raw_velocity = self.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=prefix_cache["past_key_values"],
            x_t=action_window,
            timestep=timestep_window,
            observation_delay_chunks=observation_delay_chunks,
            observation_delay_tokens=observation_delay_tokens,
            tactile_embedding=tactile_embedding,
            tactile_condition=tactile_condition,
            inference_mode="stream",
            context_cfg_lambda=context_cfg_lambda,
            apply_head_guidance=False,
        )

        head_indices = torch.arange(0, horizon, block, device=device)
        current_heads = raw_velocity[:, head_indices]
        if history_length:
            # Use only fields recorded before the current position. The
            # current ``v_h(A)`` is added after guidance is computed.
            head_history = history[:, head_indices]
            head_valid = history_valid[:, head_indices]
            history_sum = (head_history * head_valid.unsqueeze(-1).to(head_history.dtype)).sum(dim=2)
            history_count = head_valid.sum(dim=2, keepdim=True).to(current_heads.dtype)
        else:
            history_sum = torch.zeros_like(current_heads)
            history_count = torch.zeros(
                (*current_heads.shape[:2], 1), dtype=current_heads.dtype, device=device
            )

        if inference_mode == "stream-head-mean":
            count = history_count + 1.0
            guided_heads = (history_sum + current_heads) / count
        elif inference_mode == "stream-head-context-cfg":
            guided_heads = current_heads + context_cfg_lambda * (
                history_sum - history_count * current_heads
            )
        else:
            guided_heads = current_heads

        num_chunks = max(1, math.ceil(horizon / block))
        head_dt = -torch.minimum(
            timestep_window[:, head_indices],
            torch.full_like(timestep_window[:, head_indices], 1.0 / num_chunks),
        )
        action_window[:, head_indices] = action_window[:, head_indices] + head_dt[..., None] * guided_heads
        timestep_window[:, head_indices] = torch.clamp(
            timestep_window[:, head_indices] - 1.0 / num_chunks,
            min=0.0,
        )
        if history_length:
            history = torch.cat([history[:, :, 1:], raw_velocity[:, :, None, :]], dim=2)
            history_valid = torch.cat(
                [history_valid[:, :, 1:], torch.ones((bsize, horizon, 1), dtype=torch.bool, device=device)],
                dim=2,
            )
        self._stream_head_state = {
            "action_window": action_window.detach(),
            "timestep_window": timestep_window.detach(),
            "history": history.detach(),
            "history_valid": history_valid.detach(),
        }
        return action_window

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs: Unpack[RTCActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        else:
            noise = noise.to(device=device, dtype=torch.float32)

        observation_delay_chunks = kwargs.get("observation_delay_chunks")
        observation_delay_tokens = kwargs.get("observation_delay_tokens")
        # ``tactile_batch`` is an internal condition, not an RTC kwarg. Pop it
        # before forwarding the remaining kwargs, otherwise the explicit
        # argument below is passed twice to ``sample_actions_from_prefix``.
        tactile_batch = kwargs.pop("tactile_batch", None)
        prefix_cache = self.encode_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
            observation_delay_chunks=observation_delay_chunks,
            observation_delay_tokens=observation_delay_tokens,
            tactile_batch=tactile_batch,
        )
        return self.sample_actions_from_prefix(prefix_cache, noise=noise, tactile_batch=tactile_batch, **kwargs)

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        observation_delay_chunks=None,
        observation_delay_tokens=None,
        tactile_batch=None,
        tactile_embedding=None,
        tactile_condition=None,
        inference_mode="stream",
        context_cfg_lambda=0.1,
        apply_head_guidance=True,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        if tactile_embedding is None and tactile_batch is not None:
            tactile_embedding = self._encode_tactile(tactile_batch)
        if tactile_condition is None and tactile_batch is not None and self.tactile_adarms is not None:
            tactile_condition = self._encode_tactile_adarms(tactile_batch)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            x_t,
            timestep,
            observation_delay_chunks=observation_delay_chunks,
            observation_delay_tokens=observation_delay_tokens,
            tactile_embedding=tactile_embedding,
            tactile_condition=tactile_condition,
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        if tactile_embedding is not None and self.config.use_tactile_token:
            full_att_2d_masks = self._apply_tactile_token_mask(
                full_att_2d_masks,
                prefix_len=prefix_len,
                action_count=x_t.shape[1],
                query_offset=prefix_len,
            )
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        if tactile_embedding is not None and self.config.use_tactile_token:
            copy_index = 1
            action0_index = 2
            position_ids[:, action0_index] = position_ids[:, copy_index]

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
        )
        if past_key_values is not None:
            # Self-attention layers append suffix K/V in place; restore the prefix for the next step.
            for layer in past_key_values.layers:
                layer.crop(prefix_len - layer.get_seq_length())
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        if not apply_head_guidance:
            return v_t
        return self._apply_head_guidance(
            v_t,
            inference_mode=inference_mode,
            context_cfg_lambda=context_cfg_lambda,
            chunk_size=self.config.chunk_size,
            streaming_chunk_size=(
                self.config.streaming_chunk_size if self.config.streaming else self.config.chunk_size
            ),
        )

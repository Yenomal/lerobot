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

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotObservation
from lerobot.processor import (
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_smolvla import SmolVLAConfig
from .tactile_hmm import TACTILE_HMM_RAW, TACTILE_HMM_STATE


@ProcessorStepRegistry.register(name="smolvla_tactile_hmm_raw")
@dataclass
class TactileHMMRawProcessorStep(ObservationProcessorStep):
    """Keep raw markers alongside normalized inputs for direct policy inference."""

    features: tuple[str, ...]

    def observation(self, observation):
        if TACTILE_HMM_STATE not in observation and all(key in observation for key in self.features):
            for side, key in enumerate(self.features):
                observation[f"{TACTILE_HMM_RAW}.{side}"] = observation[key]
        return observation

    def get_config(self) -> dict[str, Any]:
        return {"features": list(self.features)}

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register(name="smolvla_tactile_image_size")
@dataclass
class TactileImageSizeProcessorStep(ObservationProcessorStep):
    """Scale both baseline and current marker coordinates to image units."""

    features: tuple[str, ...]

    def matches(self, key: str) -> bool:
        if key.endswith("_is_pad"):
            return False
        frame_key = key.removesuffix("_vlm_history").removesuffix("_history")
        return frame_key in self.features or (
            "tactile" in frame_key.lower() and "marker" in frame_key.lower()
        )

    def observation(self, observation: RobotObservation) -> RobotObservation:
        for key, value in observation.items():
            if not self.matches(key):
                continue
            marker = torch.as_tensor(value)
            if marker.ndim < 3 or marker.shape[-3] != 2 or marker.shape[-1] != 2:
                raise ValueError(f"Image-size tactile normalization requires [..., 2, markers, 2] for {key}")
            observation[key] = marker / marker.new_tensor((320, 240))
        return observation

    def get_config(self) -> dict[str, Any]:
        return {"features": list(self.features)}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def configure_tactile_preprocessor(config: SmolVLAConfig, preprocessor: PolicyProcessorPipeline) -> None:
    """Apply the configured marker scaling and retain raw values for the HMM."""
    metadata = config.tactile_hmm_metadata
    image_size = config.use_tactile and config.tactile_normalization == "image_size"
    hmm_keys = tuple(metadata.get("policy_features") or metadata["features"]) if metadata is not None else ()
    tactile_keys = set(config.tactile_feature_names).union(hmm_keys)
    steps = [step for step in preprocessor.steps if not isinstance(step, TactileImageSizeProcessorStep)]
    normalizer_index = next(
        (i for i, step in enumerate(steps) if isinstance(step, NormalizerProcessorStep)), len(steps)
    )
    marker_steps = []
    if (
        config.use_tactile_hmm
        and metadata is not None
        and (image_size or not config.use_tactile_delta)
        and not any(isinstance(step, TactileHMMRawProcessorStep) for step in steps)
    ):
        marker_steps.append(TactileHMMRawProcessorStep(hmm_keys))
    scale = TactileImageSizeProcessorStep(tuple(sorted(tactile_keys))) if image_size else None
    if scale is not None:
        marker_steps.append(scale)
    steps[normalizer_index:normalizer_index] = marker_steps
    preprocessor.steps = steps

    for step in steps:
        if not isinstance(step, NormalizerProcessorStep):
            continue
        selected = (
            step.features if step.normalize_observation_keys is None else step.normalize_observation_keys
        )
        if scale is not None:
            step.normalize_observation_keys = {key for key in selected if not scale.matches(key)}
        elif config.use_tactile_delta:
            # Legacy delta checkpoints were trained on raw coordinate differences.
            step.normalize_observation_keys = {
                key for key in selected if "tactile" not in key.lower() and key not in tactile_keys
            }


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Normalizing input and output features based on dataset statistics.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    steps = make_default_policy_processor_steps(config, dataset_stats)

    input_steps = [
        steps.rename_observations,  # To mimic the same processor as pretrained one
        steps.add_batch_dim,
        NewLineTaskProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        steps.to_device,
        steps.normalize,
    ]
    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]
    processors = make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
    configure_tactile_preprocessor(config, processors[0])
    return processors

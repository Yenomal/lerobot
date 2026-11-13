# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import math
from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features

from ..rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    # Research-time streaming controls shared with the Pi05 experiments.
    # ``mask`` and ``bidirectional`` use the same action-block visibility rule
    # as the reference implementation; ``causal`` advances attention at every
    # streaming chunk boundary. ``random_gate`` samples a continuous gate in
    # training and adds its log as an inter-chunk attention bias. Inference
    # uses bidirectional attention for this mode.
    streaming: bool = False
    streaming_chunk_size: int = 5
    streaming_attention_mode: str = "bidirectional"
    streaming_constant_weight: float = 0.2
    streaming_chunk_wise_weight: float = 0.8

    # Concatenate the FM timestep encoding with action embeddings before the MLP.
    # Disabling this removes the explicit time condition, not action noising.
    use_noise_embedding: bool = True
    # Pi0.5 flow-time AdaRMS: per-token scale/shift, residual gates and final norm.
    use_noise_adarms: bool = False

    # Observation-delay conditioning. The values are discrete chunk/token
    # counts and are sampled by the policy during training when the dataset
    # does not provide explicit delay fields.
    observation_delay_max_chunks: int = 0
    observation_delay_max_tokens: int = 0
    # Encode each action's age d+i with sinusoidal features followed by an MLP.
    # False preserves the original scalar/table encoding for existing checkpoints.
    use_observation_delay_sincos: bool = False
    use_observation_delay_embedding: bool = False
    # Per-expert-layer scale/shift after the attention and MLP RMSNorms.
    use_observation_delay_adarms: bool = False
    # Keep pretrained RMSNorm; add scale/shift and zero-initialized residual gates.
    use_observation_delay_adaln_zero: bool = False
    # At d=0, bypass Zero scale/shift and use residual gate=1 for every action.
    observation_delay_zero_bypass: bool = False
    # Add FM delay tokens: one shared token, or one per action with sin/cos encoding.
    use_observation_delay_token: bool = False

    # Tactile conditioning. The feature names are intentionally configurable
    # because UniVTAC datasets use both dotted and flat naming conventions.
    use_tactile: bool = False
    use_tactile_adarms: bool = False
    use_tactile_token: bool = False
    use_tactile_token_history: bool = False
    tactile_token_history_length: int = 1
    # Width of the causal tactile history encoder. The resulting token is
    # projected into the FM/action-expert stream and is refreshed per inference.
    tactile_tcn_width: int = 256
    tactile_feature_names: tuple[str, ...] = (
        "observation.tactile.left_marker",
        "observation.tactile.right_marker",
    )

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Input dimensions for the state/action projection MLPs.  SmolVLA used to
    # hard-code a 32-dimensional padded vector here, which made checkpoints
    # unnecessarily tied to one embodiment.  Keep the small default used by
    # the lightweight policy and override these explicitly for a dataset with
    # a different number of joints (for example, ``--policy.state_dim=9
    # --policy.action_dim=9``).
    state_dim: int = 6
    action_dim: int = 6

    # Legacy padding fields kept in the serialized config for compatibility
    # with older SmolVLA checkpoints.  New model code uses state_dim/action_dim.
    max_state_dim: int = 32
    max_action_dim: int = 32
    # Explicitly enable overlap-copy adaptation when loading a legacy projection
    # checkpoint whose padded 32-wide MLPs differ from the configured dimensions.
    adapt_pretrained_dimensions: bool = False

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to relative values with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10
    optimizer_fused: bool | None = None

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    # Set to False when training the expert from scratch; set to True when
    # initializing from pretrained SmolVLA weights.
    load_vlm_weights: bool = False

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    # Non-positive means that the action expert matches the VLM depth.
    num_expert_layers: int = -1
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        allowed_attention_modes = {"bidirectional", "mask", "causal", "random_gate"}
        if self.streaming_attention_mode not in allowed_attention_modes:
            raise ValueError(
                "SmolVLA streaming_attention_mode must be one of: bidirectional, mask, causal, random_gate"
            )
        if self.streaming_chunk_size <= 0:
            raise ValueError("streaming_chunk_size must be positive")
        if self.streaming and self.chunk_size % self.streaming_chunk_size != 0:
            raise ValueError("chunk_size must be divisible by streaming_chunk_size when streaming=True")
        if self.streaming_attention_mode == "random_gate" and not self.streaming:
            raise ValueError("random_gate requires streaming=True")
        if self.streaming_constant_weight < 0 or self.streaming_chunk_wise_weight < 0:
            raise ValueError("streaming regime weights must be non-negative")
        if self.streaming and self.streaming_constant_weight + self.streaming_chunk_wise_weight <= 0:
            raise ValueError("At least one streaming regime weight must be positive")
        if self.observation_delay_max_chunks < 0 or self.observation_delay_max_tokens < 0:
            raise ValueError("observation delay limits must be non-negative")
        if self.observation_delay_max_chunks and self.observation_delay_max_tokens:
            raise ValueError(
                "observation_delay_max_chunks and observation_delay_max_tokens cannot both be non-zero"
            )
        if (self.observation_delay_max_chunks or self.observation_delay_max_tokens) and not self.streaming:
            raise ValueError("observation delay requires streaming=True")
        if (
            (
                self.use_observation_delay_embedding
                or self.use_observation_delay_adarms
                or self.use_observation_delay_adaln_zero
                or self.use_observation_delay_token
            )
            and self.observation_delay_max_chunks == 0
            and self.observation_delay_max_tokens == 0
        ):
            raise ValueError("observation delay conditioning requires a non-zero observation delay limit")
        if self.use_observation_delay_adarms and self.use_observation_delay_adaln_zero:
            raise ValueError("observation delay AdaRMS and AdaLN-Zero cannot both be enabled")
        if self.observation_delay_zero_bypass and (
            not self.use_observation_delay_adaln_zero or self.use_noise_adarms
        ):
            raise ValueError("observation_delay_zero_bypass requires delay AdaLN-Zero without noise AdaRMS")
        if self.use_tactile_adarms and not self.use_tactile:
            raise ValueError("use_tactile_adarms requires use_tactile=True")
        if self.use_tactile_token and not self.use_tactile:
            raise ValueError("use_tactile_token requires use_tactile=True")
        if self.use_tactile_token_history and not self.use_tactile_token:
            raise ValueError("use_tactile_token_history requires use_tactile_token=True")
        if self.tactile_token_history_length <= 0:
            raise ValueError("tactile_token_history_length must be positive")
        if self.tactile_tcn_width <= 0:
            raise ValueError("tactile_tcn_width must be positive")
        if not math.isfinite(self.streaming_constant_weight) or not math.isfinite(
            self.streaming_chunk_wise_weight
        ):
            raise ValueError("streaming regime weights must be finite")
        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError(
                f"SmolVLA state_dim/action_dim must be positive, got {self.state_dim}/{self.action_dim}."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by SmolVLA for "
                "Aloha real models. It is not ported yet in LeRobot."
            )

    def validate_features(self) -> None:
        if self.input_features is None:
            raise ValueError("`input_features` must be resolved before `validate_features()` is called.")

        state_shape = None
        dataset_meta = getattr(self, "_runtime_dataset_meta", None)
        if dataset_meta is not None:
            dataset_state = dataset_meta.features.get("observation.state")
            if dataset_state is not None:
                state_shape = dataset_state.get("shape")
        if state_shape is None:
            state_feature = self.input_features.get("observation.state")
            state_shape = state_feature.shape if state_feature is not None else None
        if state_shape is not None and state_shape[0] > self.state_dim:
            raise ValueError(
                f"Dataset state dimension {state_shape[0]} exceeds SmolVLA state_dim={self.state_dim}. "
                "Set --policy.state_dim to the dataset dimension."
            )
        action_feature = (self.output_features or {}).get("action")
        if action_feature is not None and action_feature.shape[0] > self.action_dim:
            raise ValueError(
                f"Dataset action dimension {action_feature.shape[0]} exceeds SmolVLA action_dim={self.action_dim}. "
                "Set --policy.action_dim to the dataset dimension."
            )

        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def set_dataset_feature_metadata(self, features: dict[str, dict]) -> None:
        """Use the fine-tuning dataset's state and real camera contracts.

        A pretrained SmolVLA config already contains the base embodiment's
        features (historically 6-D state and three 256x256 cameras). Keeping
        those fields when fine-tuning another dataset makes the saved
        checkpoint advertise the wrong runtime contract. The default policy
        consumes ``observation.state`` and RGB images; when ``use_tactile`` is
        enabled, tactile marker streams are retained as additional inputs.
        """
        policy_features = dataset_to_policy_features(features)
        self.input_features = {
            key: feature
            for key, feature in policy_features.items()
            if key == OBS_STATE
            or feature.type is FeatureType.VISUAL
            or (self.use_tactile and "tactile" in key.lower())
        }

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            fused=self.optimizer_fused,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        # Training receives a contiguous current-to-past window. Runtime still
        # accepts one current observation; delayed inference is represented by
        # the age of the cached VLM prefix.
        return list(range(-(self.n_obs_steps - 1), 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

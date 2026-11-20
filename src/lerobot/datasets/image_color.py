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

"""Explicit correction of legacy BGR dataset images for RGB training."""

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class BGRToRGB:
    """Swap channels of CHW or temporal CHW images before optional augmentation.

    Dataset readers apply image transforms only to color cameras, excluding depth.
    This callable is also pickleable for DataLoader workers using spawn.
    """

    transforms: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim < 3 or image.shape[-3] != 3:
            raise ValueError(f"BGRToRGB expects [..., 3, H, W] images, got {tuple(image.shape)}")
        image = image.flip(-3)
        return image if self.transforms is None else self.transforms(image)

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

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.default import DatasetConfig
from lerobot.datasets import factory
from lerobot.datasets.image_color import BGRToRGB
from lerobot.datasets.video_utils import apply_rgb_transforms
from lerobot.transforms import ImageTransformConfig, ImageTransformsConfig
from lerobot.utils.constants import IMAGENET_STATS

pytestmark = pytest.mark.dataset


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize("temporal", [False, True])
def test_conversion_precedes_augmentation_and_preserves_depth(dtype, temporal):
    source = torch.tensor([10, 20, 30], dtype=dtype).reshape(3, 1, 1).expand(3, 4, 4)
    if temporal:
        source = source.unsqueeze(0).expand(2, -1, -1, -1)
    expected = torch.tensor([30, 20, 10], dtype=dtype)

    def augmentation(image):
        torch.testing.assert_close(image[..., 0, 0], expected.expand(image.shape[:-2]))
        return image + 1

    depth = torch.ones(1, 4, 4)
    item = {"image": source, "depth": depth, "state": torch.arange(3)}
    apply_rgb_transforms(item, BGRToRGB(augmentation), ["image", "depth"], ["depth"])
    torch.testing.assert_close(item["image"][..., 0, 0], (expected + 1).expand(source.shape[:-2]))
    assert item["depth"] is depth
    assert torch.equal(item["state"], torch.arange(3))
    assert source[..., 0, 0].reshape(-1, 3)[0].tolist() == [10, 20, 30]


@pytest.mark.parametrize("backend", ["dataloader", "gpu"])
@pytest.mark.parametrize("streaming,eval_split", [(False, 0.0), (True, 0.0), (False, 0.5)])
@pytest.mark.parametrize("imagenet_stats", [False, True])
def test_factory_corrects_train_eval_pixels_and_stats(
    monkeypatch, backend, streaming, eval_split, imagenet_stats
):
    class DummyDataset:
        def __init__(self, *args, episodes=None, image_transforms=None, **kwargs):
            self.episodes = episodes
            self.num_episodes = 2
            self.image_transforms = image_transforms
            self.meta = SimpleNamespace(
                camera_keys=["image", "depth"],
                depth_keys=["depth"],
                episodes={"tasks": [["pick"], ["pick"]]},
                stats={
                    "image": {"mean": torch.tensor([0.1, 0.2, 0.3]).reshape(3, 1, 1)},
                    "depth": {"mean": torch.tensor([42.0])},
                },
            )

    monkeypatch.setattr(factory, "LeRobotDataset", DummyDataset)
    monkeypatch.setattr(factory, "StreamingLeRobotDataset", DummyDataset)
    monkeypatch.setattr(
        factory,
        "load_dataset_metadata",
        lambda *a, **kw: SimpleNamespace(storage_format="lerobot", total_episodes=2),
    )
    monkeypatch.setattr(factory, "resolve_delta_timestamps", lambda *a, **kw: None)
    transforms = ImageTransformsConfig(
        enable=True,
        backend=backend,
        tfs={"invert": ImageTransformConfig(type="RandomInvert", kwargs={"p": 1.0})},
    )
    cfg = SimpleNamespace(
        dataset=DatasetConfig(
            repo_id="test/colors",
            image_color_order="bgr",
            image_transforms=transforms,
            streaming=streaming,
            eval_split=eval_split,
            use_imagenet_stats=imagenet_stats,
        ),
        trainable_config=object(),
        seed=0,
        num_workers=0,
        tolerance_s=1e-4,
        rename_map={},
    )
    train, evaluation = factory.make_train_eval_datasets(cfg)
    pixels = torch.tensor([0.1, 0.2, 0.3]).reshape(3, 1, 1).expand(3, 4, 4)
    rgb = torch.tensor([0.3, 0.2, 0.1])
    torch.testing.assert_close(
        train.image_transforms(pixels)[:, 0, 0], 1 - rgb if backend == "dataloader" else rgb
    )
    datasets = [train]
    if eval_split:
        torch.testing.assert_close(evaluation.image_transforms(pixels)[:, 0, 0], rgb)
        datasets.append(evaluation)
    else:
        assert evaluation is None
    for dataset in datasets:
        expected_mean = torch.tensor(IMAGENET_STATS["mean"]).flatten() if imagenet_stats else rgb
        torch.testing.assert_close(dataset.meta.stats["image"]["mean"].flatten(), expected_mean)
        assert dataset.meta.stats["depth"]["mean"].item() == 42


def test_default_rgb_and_invalid_declaration():
    cfg = SimpleNamespace(dataset=DatasetConfig(repo_id="test/colors"))
    assert cfg.dataset.image_color_order == "rgb"
    assert factory._make_training_image_transforms(cfg) is None
    with pytest.raises(ValueError, match="image_color_order"):
        DatasetConfig(repo_id="test/colors", image_color_order="auto")

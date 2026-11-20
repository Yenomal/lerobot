#!/usr/bin/env python

"""Regression coverage for converting v2.1 Arrow extension columns."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from datasets import Array3D, Dataset, Features, Value

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.io_utils import load_stats
from lerobot.processor import NormalizerProcessorStep
from lerobot.scripts.convert_dataset_v21_to_v30 import (
    concat_data_files,
    convert_episodes_metadata,
    legacy_load_episodes_stats,
)

pytestmark = pytest.mark.dataset


def test_concat_v21_array3d_parquet_files(tmp_path):
    features = Features({"state": Array3D(shape=(2, 3, 2), dtype="float32"), "episode": Value("int64")})
    parquet_paths = []
    for episode in range(2):
        dataset = Dataset.from_dict(
            {
                "state": [[[[episode, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]]],
                "episode": [episode],
            },
            features=features,
        )
        parquet_path = tmp_path / f"episode_{episode}.parquet"
        dataset.to_parquet(parquet_path)
        parquet_paths.append(parquet_path)

    output_root = tmp_path / "converted"
    concat_data_files(parquet_paths, output_root, chunk_idx=0, file_idx=0, image_keys=[])
    output = output_root / "data/chunk-000/file-000.parquet"

    table = pq.read_table(output)
    assert table.num_rows == 2
    assert table.column("episode").to_pylist() == [0, 1]
    assert table.column("state").to_pylist()[1][0][0][0] == 1


@pytest.mark.parametrize("stored_markers", [63, 1200])
def test_convert_cropped_tactile_stats_and_normalize_history(tmp_path, stored_markers):
    source = tmp_path / "source"
    meta = source / "meta"
    meta.mkdir(parents=True)
    keys = [f"observation.tactile.{side}_marker" for side in ("left", "right")]
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "fps": 30,
                "features": {key: {"dtype": "float32", "shape": [2, 63, 2]} for key in keys},
            }
        )
    )
    rng = np.random.default_rng(0)
    frames = {key: [] for key in keys}
    legacy_episodes = []
    episode_stats = []
    episode_locations = []
    offset = 0
    for episode_index, length in enumerate([2, 3]):
        stats = {}
        for key in keys:
            values = rng.normal(size=(length, 2, stored_markers, 2))
            frames[key].append(values[:, :, :63, :])
            stats[key] = {
                stat: fn(values, axis=0).tolist()
                for stat, fn in [("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std)]
            }
            stats[key]["count"] = [length]
        # Non-tactile statistics must not be cropped.
        stats["observation.state"] = {
            "min": [0.0] * 9,
            "max": [1.0] * 9,
            "mean": [0.5] * 9,
            "std": [0.5] * 9,
            "count": [length],
        }
        legacy_episodes.append({"episode_index": episode_index, "length": length, "tasks": ["test"]})
        episode_stats.append({"episode_index": episode_index, "stats": stats})
        episode_locations.append(
            {
                "episode_index": episode_index,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": offset,
                "dataset_to_index": offset + length,
            }
        )
        offset += length
    for filename, records in [("episodes.jsonl", legacy_episodes), ("episodes_stats.jsonl", episode_stats)]:
        (meta / filename).write_text("".join(json.dumps(record) + "\n" for record in records))

    target = tmp_path / "converted"
    convert_episodes_metadata(source, target, episode_locations)
    converted_stats = load_stats(target)
    converted_episodes = pq.read_table(target / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    normalizer = NormalizerProcessorStep(
        features={key: PolicyFeature(type=FeatureType.STATE, shape=(2, 63, 2)) for key in keys},
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
        stats=converted_stats,
    )
    for key in keys:
        all_frames = np.concatenate(frames[key])
        for stat, fn in [("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std)]:
            np.testing.assert_allclose(converted_stats[key][stat], fn(all_frames, axis=0), atol=1e-12)
            for episode in converted_episodes:
                assert np.asarray(episode[f"stats/{key}/{stat}"]).shape == (2, 63, 2)
        np.testing.assert_array_equal(converted_stats[key]["count"], [5])
        history = torch.from_numpy(all_frames[:1]).float().unsqueeze(1).expand(-1, 21, -1, -1, -1)
        normalized = normalizer._normalize_observation({key: history}, inverse=False)[key]
        expected = (history - torch.tensor(all_frames.mean(axis=0))) / (
            torch.tensor(all_frames.std(axis=0)) + normalizer.eps
        )
        torch.testing.assert_close(normalized.double(), expected)
    assert converted_stats["observation.state"]["mean"].shape == (9,)
    # Conversion preserves the original metadata.
    original = json.loads((meta / "episodes_stats.jsonl").read_text().splitlines()[0])
    assert np.asarray(original["stats"][keys[0]]["mean"]).shape == (2, stored_markers, 2)


def test_legacy_tactile_stats_cannot_be_expanded(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    key = "observation.tactile.left_marker"
    (meta / "info.json").write_text(
        json.dumps({"codebase_version": "v2.1", "fps": 30, "features": {key: {"shape": [2, 63, 2]}}})
    )
    (meta / "episodes_stats.jsonl").write_text(
        json.dumps({"episode_index": 0, "stats": {key: {"mean": np.zeros((2, 62, 2)).tolist()}}}) + "\n"
    )
    with pytest.raises(ValueError, match="cannot be cropped along the marker axis"):
        legacy_load_episodes_stats(tmp_path)

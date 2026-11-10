#!/usr/bin/env python

"""Regression coverage for converting v2.1 Arrow extension columns."""

import pyarrow.parquet as pq
import pytest
from datasets import Array3D, Dataset, Features, Value

from lerobot.scripts.convert_dataset_v21_to_v30 import concat_data_files

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

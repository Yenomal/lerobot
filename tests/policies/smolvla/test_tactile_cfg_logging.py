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

import csv
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.smolvla import tactile_cfg_logging
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
from lerobot.policies.smolvla.tactile_cfg_logging import TactileCFGLambdaLogger


@pytest.mark.parametrize(
    "filename,payload",
    [
        ("resolved_experiment.yaml", "exp_name: '20261116_005'"),
        ("experiment.yaml", "exp_name: '20261116_005'\nextends: missing_base.yaml"),
        ("config.json", json.dumps({"exp_name": "20261116_005"})),
        ("train_config.json", json.dumps({"job_name": "20261116_005"})),
    ],
)
def test_log_reads_experiment_from_saved_checkpoint_and_flushes_each_step(
    tmp_path, monkeypatch, filename, payload
):
    root = tmp_path / "logs"
    monkeypatch.setattr(tactile_cfg_logging, "DEFAULT_LOG_ROOT", root)
    checkpoint = tmp_path / "pretrained_model_ema"
    checkpoint.mkdir()
    (checkpoint / filename).write_text(payload)
    trace = TactileCFGLambdaLogger(checkpoint)
    assert not root.exists()
    for _ in range(2):
        trace.record(
            inference_mode="stream-head-tactile-cfg",
            source="learned",
            lambda_values=[0.25, 1.75],
            timesteps=[0.1, 0.2],
        )
    trace.reset_session()
    trace.record(
        inference_mode="stream-head-tactile-cfg", source="fixed", lambda_values=[1.5], timesteps=[0.3]
    )
    files = list((root / "20261116_005").glob("*.csv"))
    assert len(files) == 1
    with files[0].open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert [row["step"] for row in rows] == ["0", "0", "1", "1", "0"]
    assert [row["session"] for row in rows] == ["0", "0", "0", "0", "1"]
    assert [float(row["lambda"]) for row in rows] == [0.25, 1.75, 0.25, 1.75, 1.5]
    assert [row["batch_index"] for row in rows] == ["0", "1", "0", "1", "0"]
    assert all(row["checkpoint"] == str(checkpoint.resolve()) for row in rows)
    assert all(row["timestamp_utc"].endswith("+00:00") for row in rows)
    # Another model/process must not append to the first trace.
    another = TactileCFGLambdaLogger(checkpoint)
    another.record(
        inference_mode="stream-head-tactile-cfg", source="fixed", lambda_values=[0.5], timesteps=[0.1]
    )
    assert another.path != trace.path


@pytest.mark.parametrize("learned", [False, True])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize(
    "mode", ["stream-head-tactile-cfg", "stream-head-mean-tactile-cfg", "stream-head-tactile", "stream-head"]
)
def test_only_applied_cfg_strength_is_logged_with_pre_integration_time(
    tmp_path, monkeypatch, learned, mode, padded
):
    root = tmp_path / "logs"
    monkeypatch.setattr(tactile_cfg_logging, "DEFAULT_LOG_ROOT", root)
    checkpoint = tmp_path / "weights"
    checkpoint.mkdir()
    (checkpoint / "resolved_experiment.yaml").write_text("exp_name: '20261116_005'")
    (checkpoint / "experiment.yaml").write_text("exp_name: 'wrong'")
    model = object.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        use_tactile_cfg_gate=learned,
        use_fm_tactile_token=True,
        chunk_size=2,
        streaming=True,
        streaming_chunk_size=2,
        action_dim=2 if padded else 1,
        action_feature=SimpleNamespace(shape=(1,)),
        pretrained_path="wrong_base_model",
    )
    trace = model._tactile_cfg_lambda_logger = TactileCFGLambdaLogger(checkpoint)
    model._exp_random_tactile = learned
    model.tactile_adarms = None
    model._encode_tactile = lambda batch: torch.ones(2, 1)
    width = model.config.action_dim
    raw = torch.full((2, 2, width), 2.0)
    tactile = torch.full((2, width), 10.0)
    if padded:
        raw[..., 1] = 1000.0
        tactile[..., 1] = -1000.0
    strength = torch.tensor([[0.25], [1.75]])

    def denoise(**kwargs):
        if kwargs.get("return_tactile_lambda"):
            return raw, tactile, strength
        if kwargs.get("return_tactile_copy"):
            return raw, tactile
        return raw

    model.denoise_step = denoise
    model.sample_noise = lambda shape, device: torch.zeros(shape)
    model._stream_head_state = {
        "action_window": torch.zeros(2, 2, width),
        "timestep_window": torch.full((2, 2), 0.4),
        "history": torch.full((2, 2, 1, width), 6.0),
        "history_valid": torch.ones(2, 2, 1, dtype=torch.bool),
    }
    prefix = {"prefix_pad_masks": torch.ones(2, 1, dtype=torch.bool), "past_key_values": None}
    actions = model.stream_head_step_from_prefix(
        prefix, tactile_batch={}, inference_mode=mode, tactile_cfg_lambda=1.5
    )
    if mode in ("stream-head", "stream-head-tactile"):
        assert not root.exists()
        return
    assert trace.path.parent == root / "20261116_005"
    with trace.path.open() as stream:
        rows = list(csv.DictReader(stream))
    expected_strength = [0.25, 1.75] if learned else [1.5, 1.5]
    assert [float(row["lambda"]) for row in rows] == expected_strength
    assert [row["source"] for row in rows] == ["learned" if learned else "fixed"] * 2
    assert [row["exp_random_tactile"] for row in rows] == [str(learned)] * 2
    assert all(float(row["timestep"]) == pytest.approx(0.4) for row in rows)
    assert [float(row["ca0_a0_mse"]) for row in rows] == [64.0, 64.0]
    assert [float(row["ca0_a0_rms"]) for row in rows] == [8.0, 8.0]
    assert [float(row["a0_rms"]) for row in rows] == [2.0, 2.0]
    assert [float(row["ca0_rms"]) for row in rows] == [10.0, 10.0]
    assert [float(row["ca0_a0_cosine_similarity"]) for row in rows] == [1.0, 1.0]
    assert [float(row["cfg_correction_rms"]) for row in rows] == [value * 8 for value in expected_strength]
    base = 4.0 if mode == "stream-head-mean-tactile-cfg" else 2.0
    torch.testing.assert_close(actions[:, 0, 0], -0.4 * (base + torch.tensor(expected_strength) * 8))
    model.reset_stream_head_state()
    assert trace.session == 1 and trace.step == 0


def test_missing_or_unsafe_experiment_name_falls_back_to_unknown(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(tactile_cfg_logging, "DEFAULT_LOG_ROOT", tmp_path / "logs")
    (tmp_path / "config.json").write_text(json.dumps({"exp_name": "../escape"}))
    trace = TactileCFGLambdaLogger(tmp_path)
    trace.record(
        inference_mode="stream-head-tactile-cfg", source="fixed", lambda_values=[1.5], timesteps=[0.1]
    )
    assert trace.path.parent == tmp_path / "logs" / "unknown"
    assert "no experiment name" in caplog.text


def test_logging_failure_does_not_interrupt_inference(tmp_path, monkeypatch, caplog):
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("occupied")
    monkeypatch.setattr(tactile_cfg_logging, "DEFAULT_LOG_ROOT", not_a_directory)
    trace = TactileCFGLambdaLogger(tmp_path)
    for _ in range(2):
        trace.record(
            inference_mode="stream-head-tactile-cfg", source="fixed", lambda_values=[1.5], timesteps=[0.1]
        )
    assert caplog.text.count("logging disabled for this model") == 1


@pytest.mark.parametrize(
    "raw,tactile,expected",
    [
        ([1.0, 0.0], [2.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 2.0], 0.0),
        ([1.0, 0.0], [-2.0, 0.0], -1.0),
        ([3.0, 4.0], [4.0, 3.0], 0.96),
        ([0.0, 0.0], [1.0, 0.0], None),
        ([1.0, 0.0], [0.0, 0.0], None),
        ([1e-10, 0.0], [1.0, 0.0], None),
    ],
)
def test_logged_cosine_covers_directions_and_undefined_zero_fields(
    tmp_path, monkeypatch, raw, tactile, expected
):
    monkeypatch.setattr(tactile_cfg_logging, "DEFAULT_LOG_ROOT", tmp_path / "logs")
    (tmp_path / "config.json").write_text(json.dumps({"exp_name": "cosine_test"}))
    model = object.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        use_fm_tactile_token=True,
        use_tactile_cfg_gate=False,
        chunk_size=1,
        streaming=False,
        action_dim=3,
        action_feature=SimpleNamespace(shape=(2,)),
    )
    model._tactile_cfg_lambda_logger = TactileCFGLambdaLogger(tmp_path)
    model.tactile_adarms = None
    model._encode_tactile = lambda batch: torch.ones(1, 1)
    # Opposite large padding values must not affect angular similarity.
    raw_field = torch.tensor([[raw + [1000.0]]])
    tactile_field = torch.tensor([tactile + [-1000.0]])
    model.denoise_step = lambda **kwargs: (raw_field, tactile_field)
    model.sample_noise = lambda shape, device: torch.zeros(shape)
    model._stream_head_state = {
        "action_window": torch.zeros(1, 1, 3),
        "timestep_window": torch.ones(1, 1),
        "history": torch.zeros(1, 1, 0, 3),
        "history_valid": torch.zeros(1, 1, 0, dtype=torch.bool),
    }
    prefix = {"prefix_pad_masks": torch.ones(1, 1, dtype=torch.bool), "past_key_values": None}
    model.stream_head_step_from_prefix(prefix, tactile_batch={}, inference_mode="stream-head-tactile-cfg")
    with model._tactile_cfg_lambda_logger.path.open() as stream:
        value = next(csv.DictReader(stream))["ca0_a0_cosine_similarity"]
    if expected is None:
        assert value == ""
    else:
        assert float(value) == pytest.approx(expected, abs=1e-6)

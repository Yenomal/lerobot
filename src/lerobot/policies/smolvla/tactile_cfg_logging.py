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

"""CSV traces of the tactile CFG strength actually applied during inference."""

import csv
import json
import logging
import math
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml

DEFAULT_LOG_ROOT = Path(__file__).resolve().parents[4] / "log" / "tactile-cfg-lambda"
logger = logging.getLogger(__name__)


def _checkpoint_exp_name(checkpoint: Path) -> str:
    """Read saved metadata, never the original/base policy's pretrained_path."""
    for filename in ("resolved_experiment.yaml", "experiment.yaml", "config.json", "train_config.json"):
        path = checkpoint / filename
        if not path.is_file():
            continue
        try:
            with path.open() as stream:
                config = json.load(stream) if path.suffix == ".json" else yaml.safe_load(stream)
        except (OSError, ValueError, yaml.YAMLError):
            logger.warning("Could not read tactile CFG experiment metadata from %s", path, exc_info=True)
            continue
        if not isinstance(config, dict):
            continue
        name = config.get("exp_name")
        if name is None and filename == "train_config.json":
            name = config.get("job_name")
        if (
            isinstance(name, str)
            and name.strip()
            and name not in (".", "..")
            and "/" not in name
            and "\\" not in name
        ):
            return name
    logger.warning("Checkpoint %s has no experiment name; tactile CFG logs use 'unknown'", checkpoint)
    return "unknown"


class TactileCFGLambdaLogger:
    """One file per loaded model/process, created only on a real CFG step.

    Files are closed after each write, so rows are visible immediately and the
    model owns no open handles that could break worker spawning/deepcopy.
    """

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint.resolve()
        self.path: Path | None = None
        self.session = 0
        self.step = 0
        self._pid: int | None = None
        self._failed = False

    def reset_session(self) -> None:
        self.session += 1
        self.step = 0

    def record(
        self,
        *,
        inference_mode: str,
        source: str,
        lambda_values: Sequence[float],
        timesteps: Sequence[float],
        branch_mse: Sequence[float] | None = None,
        a0_rms: Sequence[float] | None = None,
        ca0_rms: Sequence[float] | None = None,
        cosine_similarity: Sequence[float] | None = None,
        exp_random_tactile: bool = False,
    ) -> None:
        if self._failed:
            return
        timestamp = datetime.now(UTC)
        try:
            new_file = self.path is None or self._pid != os.getpid()
            if new_file:
                directory = DEFAULT_LOG_ROOT / _checkpoint_exp_name(self.checkpoint)
                directory.mkdir(parents=True, exist_ok=True)
                self._pid = os.getpid()
                self.path = directory / f"{timestamp:%Y%m%dT%H%M%S%fZ}_{self._pid}_{uuid4().hex[:8]}.csv"
            assert self.path is not None
            with self.path.open("x" if new_file else "a", newline="") as stream:
                writer = csv.writer(stream)
                if new_file:
                    writer.writerow(
                        (
                            "timestamp_utc",
                            "session",
                            "step",
                            "batch_index",
                            "inference_mode",
                            "source",
                            "lambda",
                            "timestep",
                            "checkpoint",
                            "ca0_a0_mse",
                            "ca0_a0_rms",
                            "a0_rms",
                            "ca0_rms",
                            "cfg_correction_rms",
                            "ca0_a0_cosine_similarity",
                            "exp_random_tactile",
                        )
                    )
                for index, (strength, timestep) in enumerate(zip(lambda_values, timesteps, strict=True)):
                    writer.writerow(
                        (
                            timestamp.isoformat(),
                            self.session,
                            self.step,
                            index,
                            inference_mode,
                            source,
                            strength,
                            timestep,
                            str(self.checkpoint),
                            branch_mse[index] if branch_mse is not None else "",
                            branch_mse[index] ** 0.5 if branch_mse is not None else "",
                            a0_rms[index] if a0_rms is not None else "",
                            ca0_rms[index] if ca0_rms is not None else "",
                            abs(strength) * branch_mse[index] ** 0.5 if branch_mse is not None else "",
                            cosine_similarity[index]
                            if cosine_similarity is not None and math.isfinite(cosine_similarity[index])
                            else "",
                            exp_random_tactile,
                        )
                    )
            if new_file:
                logger.info("Recording tactile CFG lambda to %s", self.path)
            self.step += 1
        except OSError:
            self._failed = True
            logger.warning(
                "Could not write tactile CFG lambda log; logging disabled for this model", exc_info=True
            )

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

"""Layered YAML experiment configs mapped into LeRobot dataclass configs."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml

SCHEMA_VERSION = 1
EXPERIMENT_SOURCE_FILENAME = "experiment.yaml"
EXPERIMENT_RESOLVED_FILENAME = "resolved_experiment.yaml"
EXPERIMENT_FILENAME_PATTERN = re.compile(r"^\d{8}_\d{3}$")


class ExperimentConfigError(ValueError):
    """Raised when a layered experiment YAML cannot be resolved."""


@dataclass(frozen=True)
class ExperimentConfigMetadata:
    """The original and inherited YAML sources kept beside a training checkpoint."""

    source_path: str
    source_bytes: bytes
    resolved_bytes: bytes


@dataclass(frozen=True)
class ResolvedExperimentConfig:
    """YAML payload for Draccus plus provenance for checkpoint persistence."""

    config: dict[str, Any]
    metadata: ExperimentConfigMetadata


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ExperimentConfigError(f"Experiment config must be YAML: {path}")
    try:
        value = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ExperimentConfigError(f"Invalid experiment YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"Experiment config root must be a mapping: {path}")
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge maps; replace lists, scalars, nulls and changed choices."""
    result = {
        key: _deep_merge(value, {}) if isinstance(value, Mapping) else copy.deepcopy(value)
        for key, value in base.items()
    }
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            if "type" in value and value["type"] != result[key].get("type"):
                result[key] = _deep_merge({}, value)
            else:
                result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_file(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ExperimentConfigError(f"Experiment config inheritance cycle: {chain}")

    current = _load_mapping(path)
    parent = current.pop("extends", None)
    if parent is None:
        return current
    if not isinstance(parent, str) or not parent:
        raise ExperimentConfigError(f"extends must be a non-empty relative path: {path}")
    parent_path = Path(parent)
    if parent_path.is_absolute():
        raise ExperimentConfigError(f"extends must be relative to {path}: {parent}")
    return _deep_merge(_resolve_file(path.parent / parent_path, (*stack, path)), current)


def resolve_experiment_config(path: str | Path) -> ResolvedExperimentConfig | None:
    """Resolve YAML inheritance; return ``None`` for ordinary non-layered YAML.

    The experiment-only ``schema_version`` is removed from the config passed to
    Draccus. All remaining fields are decoded by the existing LeRobot config
    dataclasses, so policy choices and their validation stay native to LeRobot.
    """
    source_path = Path(path).expanduser().resolve()
    source = _load_mapping(source_path)
    if "extends" not in source and "schema_version" not in source:
        return None

    resolved = _resolve_file(source_path)
    version = resolved.get("schema_version")
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise ExperimentConfigError(
            f"Unsupported schema_version {version!r}; expected {SCHEMA_VERSION}."
        )

    if not EXPERIMENT_FILENAME_PATTERN.fullmatch(source_path.stem):
        raise ExperimentConfigError(
            f"A runnable experiment file must be named YYYYMMDD_id.yaml, got {source_path.name!r}."
        )

    exp_name = resolved.get("exp_name") or source_path.stem
    if not isinstance(exp_name, str) or not exp_name:
        raise ExperimentConfigError("exp_name must be a non-empty string.")
    if exp_name != source_path.stem:
        raise ExperimentConfigError(
            f"exp_name {exp_name!r} must match the dated experiment filename {source_path.stem!r}."
        )

    dataset = resolved.get("dataset")
    if dataset is None:
        dataset = {}
        resolved["dataset"] = dataset
    if not isinstance(dataset, dict):
        raise ExperimentConfigError("dataset must be a mapping in a runnable experiment.")
    dataset_repo_id = dataset.get("repo_id")
    repo_id = resolved.get("repo_id") or dataset_repo_id
    if not isinstance(repo_id, str) or not repo_id:
        raise ExperimentConfigError("repo_id must be set in the experiment or its task base.")
    if dataset_repo_id is not None and dataset_repo_id != repo_id:
        raise ExperimentConfigError(
            f"Experiment repo_id {repo_id!r} differs from dataset.repo_id {dataset_repo_id!r}."
        )
    repo_path = Path(repo_id)
    if repo_path.is_absolute() or ".." in repo_path.parts:
        raise ExperimentConfigError(f"repo_id must be a relative repository path, got {repo_id!r}.")
    dataset["repo_id"] = repo_id
    resolved["repo_id"] = repo_id
    resolved["exp_name"] = exp_name

    scripts = resolved.get("scripts", "")
    if scripts is not None and not isinstance(scripts, str):
        raise ExperimentConfigError("scripts must be descriptive text or null.")
    resolved["scripts"] = scripts or ""

    output_base_dir = resolved.get("output_base_dir", "outputs/train")
    if not isinstance(output_base_dir, str) or not output_base_dir:
        raise ExperimentConfigError("output_base_dir must be a non-empty path string.")
    resolved["output_base_dir"] = output_base_dir
    if resolved.get("output_dir") is None:
        output_dir = Path(output_base_dir) / repo_path / exp_name
        resolved["output_dir"] = str(output_dir)
    if resolved.get("job_name") is None:
        resolved["job_name"] = exp_name

    resolved_bytes = yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True).encode("utf-8")
    config = copy.deepcopy(resolved)
    for metadata_key in ("schema_version", "repo_id", "exp_name", "scripts", "output_base_dir"):
        config.pop(metadata_key, None)
    metadata = ExperimentConfigMetadata(
        source_path=str(source_path),
        source_bytes=source_path.read_bytes(),
        resolved_bytes=resolved_bytes,
    )
    return ResolvedExperimentConfig(config=config, metadata=metadata)


def save_experiment_metadata(metadata: ExperimentConfigMetadata, directory: str | Path) -> None:
    """Persist source and resolved experiment YAML beside a LeRobot checkpoint config."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / EXPERIMENT_SOURCE_FILENAME).write_bytes(metadata.source_bytes)
    (directory / EXPERIMENT_RESOLVED_FILENAME).write_bytes(metadata.resolved_bytes)


def load_experiment_metadata(directory: str | Path) -> ExperimentConfigMetadata | None:
    """Restore experiment provenance from a framework checkpoint directory."""
    directory = Path(directory)
    source = directory / EXPERIMENT_SOURCE_FILENAME
    resolved = directory / EXPERIMENT_RESOLVED_FILENAME
    if not source.is_file() or not resolved.is_file():
        return None
    return ExperimentConfigMetadata(
        source_path=str(source),
        source_bytes=source.read_bytes(),
        resolved_bytes=resolved.read_bytes(),
    )

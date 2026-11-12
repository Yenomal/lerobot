# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Layered YAML experiment configs are decoded by LeRobot's native config types."""

import dataclasses
import sys
from pathlib import Path

import draccus
import pytest
import yaml

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.experiment import ExperimentConfigError, resolve_experiment_config
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401


def _write_experiment_tree(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        """schema_version: 1
dataset:
  repo_id: owner/univtac
  root: /tmp/univtac
policy:
  path: lerobot/smolvla_base
  state_dim: 6
  action_dim: 6
  adapt_pretrained_dimensions: true
batch_size: 32
num_workers: 16
steps: 4000
rename_map:
  observation.images.head: observation.images.camera1
"""
    )
    experiment = tmp_path / "20261109_001.yaml"
    experiment.write_text(
        """schema_version: 1
extends: base.yaml
exp_name: "20261109_001"
scripts: test description only
steps: 8000
policy:
  state_dim: 9
  action_dim: 9
"""
    )
    return base, experiment


def test_global_default_covers_train_pipeline_and_dataset_defaults():
    defaults = yaml.safe_load(Path("configs/default.yaml").read_text())

    train_fields = {field.name for field in dataclasses.fields(TrainPipelineConfig) if field.init}
    dataset_fields = {field.name for field in dataclasses.fields(DatasetConfig)}
    experiment_metadata_fields = {"schema_version", "repo_id", "exp_name", "scripts", "output_base_dir"}
    assert set(defaults) - experiment_metadata_fields == train_fields
    assert set(defaults["dataset"]) == dataset_fields - {"repo_id"}
    assert defaults["policy"]["type"] == "smolvla"
    assert defaults["policy"]["streaming"] is False
    assert defaults["policy"]["streaming_chunk_size"] == 5
    assert defaults["policy"]["streaming_attention_mode"] == "bidirectional"
    assert defaults["policy"]["use_tactile"] is False
    assert "n_action_steps" not in defaults["policy"]


def test_resolve_experiment_inheritance_recursively_overrides_scalars(tmp_path):
    _base, experiment = _write_experiment_tree(tmp_path)

    resolved = resolve_experiment_config(experiment)

    assert resolved is not None
    assert resolved.config["dataset"]["repo_id"] == "owner/univtac"
    assert resolved.config["steps"] == 8000
    assert resolved.config["batch_size"] == 32
    assert resolved.config["output_dir"] == "outputs/train/owner/univtac/20261109_001"
    assert resolved.config["job_name"] == "20261109_001"
    assert "scripts" not in resolved.config
    assert resolved.config["policy"]["path"] == "lerobot/smolvla_base"
    assert resolved.config["policy"]["state_dim"] == 9
    assert resolved.config["policy"]["action_dim"] == 9
    assert resolved.config["policy"]["adapt_pretrained_dimensions"] is True
    assert b"extends:" not in resolved.metadata.resolved_bytes
    assert b"scripts: test description only" in resolved.metadata.resolved_bytes
    assert resolved.metadata.source_bytes == experiment.read_bytes()


def test_train_config_from_yaml_maps_into_native_dataclasses_and_cli_wins(tmp_path, monkeypatch):
    _base, experiment = _write_experiment_tree(tmp_path)
    monkeypatch.setattr(parser, "_config_path_args", {})
    monkeypatch.setattr(parser, "_config_yaml_overrides", {})
    monkeypatch.setattr(parser, "_experiment_config_metadata", {})

    config = TrainPipelineConfig.from_pretrained(experiment, cli_args=["--steps=12000"])

    assert config.dataset.repo_id == "owner/univtac"
    assert config.dataset.root == "/tmp/univtac"
    assert config.batch_size == 32
    assert config.num_workers == 16
    assert config.steps == 12000
    assert config.output_dir.as_posix().endswith("outputs/train/owner/univtac/20261109_001")
    assert config.job_name == "20261109_001"
    assert not hasattr(config, "scripts")
    assert config.policy is None  # policy.path is loaded by TrainPipelineConfig.validate()
    assert parser.get_path_arg("policy", args=[]) == "lerobot/smolvla_base"
    assert "--state_dim=9" in parser.get_yaml_overrides("policy")
    assert "--action_dim=9" in parser.get_yaml_overrides("policy")
    assert config._experiment_config_metadata.source_bytes == experiment.read_bytes()


def test_train_entrypoint_parser_accepts_layered_yaml_and_cli_overrides(tmp_path, monkeypatch):
    _base, experiment = _write_experiment_tree(tmp_path)
    monkeypatch.setattr(parser, "_config_path_args", {})
    monkeypatch.setattr(parser, "_config_yaml_overrides", {})
    monkeypatch.setattr(parser, "_experiment_config_metadata", {})
    monkeypatch.setattr(sys, "argv", ["lerobot-train", f"--config_path={experiment}", "--steps=12000"])

    def return_config(config: TrainPipelineConfig) -> TrainPipelineConfig:
        return config

    config = parser.wrap()(return_config)()

    assert config.steps == 12000
    assert config.dataset.repo_id == "owner/univtac"
    assert parser.get_path_arg("policy", args=[]) == "lerobot/smolvla_base"
    assert config._experiment_config_metadata.source_bytes == experiment.read_bytes()


def test_yaml_policy_overrides_resolve_to_registered_policy_dataclass(tmp_path, monkeypatch):
    _base, experiment = _write_experiment_tree(tmp_path)
    monkeypatch.setattr(parser, "_config_path_args", {})
    monkeypatch.setattr(parser, "_config_yaml_overrides", {})
    monkeypatch.setattr(parser, "_experiment_config_metadata", {})

    def fake_from_pretrained(cls, model_id, *, revision=None, cli_overrides=None):
        del cls, revision
        policy_config = draccus.parse(SmolVLAConfig, args=cli_overrides or [])
        policy_config.pretrained_path = Path(model_id)
        return policy_config

    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", classmethod(fake_from_pretrained))
    config = TrainPipelineConfig.from_pretrained(experiment)
    config._resolve_pretrained_from_cli()

    assert isinstance(config.policy, SmolVLAConfig)
    assert config.policy.state_dim == 9
    assert config.policy.action_dim == 9
    assert config.policy.adapt_pretrained_dimensions is True


def test_checkpoint_saves_original_and_resolved_experiment_yaml_and_restores_them(tmp_path, monkeypatch):
    _base, experiment = _write_experiment_tree(tmp_path)
    monkeypatch.setattr(parser, "_config_path_args", {})
    monkeypatch.setattr(parser, "_config_yaml_overrides", {})
    monkeypatch.setattr(parser, "_experiment_config_metadata", {})
    config = TrainPipelineConfig.from_pretrained(experiment)

    checkpoint_config = tmp_path / "pretrained_model"
    checkpoint_config.mkdir()
    config._save_pretrained(checkpoint_config)

    assert (checkpoint_config / "experiment.yaml").read_bytes() == experiment.read_bytes()
    assert b"steps: 8000" in (checkpoint_config / "resolved_experiment.yaml").read_bytes()
    restored = TrainPipelineConfig.from_pretrained(checkpoint_config)
    assert restored._experiment_config_metadata.source_bytes == experiment.read_bytes()


def test_yaml_experiment_cannot_be_used_as_resume_checkpoint(tmp_path, monkeypatch):
    _base, experiment = _write_experiment_tree(tmp_path)
    monkeypatch.setattr(parser, "_config_path_args", {})
    monkeypatch.setattr(parser, "_config_yaml_overrides", {})
    monkeypatch.setattr(parser, "_experiment_config_metadata", {})
    experiment.write_text(experiment.read_text() + "resume: true\n")

    with pytest.raises(ValueError, match="A YAML config starts a new run"):
        TrainPipelineConfig.from_pretrained(experiment)


def test_resolver_rejects_inheritance_cycles_and_unsupported_schema(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("schema_version: 1\nextends: b.yaml\n")
    b.write_text("schema_version: 1\nextends: a.yaml\n")
    with pytest.raises(ExperimentConfigError, match="inheritance cycle"):
        resolve_experiment_config(a)

    unsupported = tmp_path / "unsupported.yaml"
    unsupported.write_text("schema_version: 2\ndataset: {}\n")
    with pytest.raises(ExperimentConfigError, match="schema_version"):
        resolve_experiment_config(unsupported)

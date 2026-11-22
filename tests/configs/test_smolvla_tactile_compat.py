"""Legacy tactile names preserve behavior without leaking into new saved configs."""

import json

import pytest
import yaml

from lerobot.configs import parser
from lerobot.configs.experiment import resolve_experiment_config
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.smolvla_compat import (
    TACTILE_CONFIG_ALIASES,
    migrate_tactile_cli_args,
    migrate_tactile_policy_fields,
)
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_checkpoint_conversion_and_new_only_save(tmp_path, enabled):
    saved = {
        "type": "smolvla",
        "device": "cpu",
        "use_tactile": True,
        "use_vlm_tactile_token": True,
        "use_tactile_token": True,
        "use_tactile_token_history": enabled,
        "fm_tactile_token_history_length": 5,
        "vlm_tactile_token_history_length": 5,
        "vlm_tactile_mask_state": enabled,
        "state_reads_vlm_tactile": enabled,
        "vlm_no_tactile_feedback": enabled,
        "fm_tactile_no_vlm_tactile_attention": enabled,
    }
    (tmp_path / "config.json").write_text(json.dumps(saved))
    config = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(config, SmolVLAConfig)
    assert config.use_fm_tactile_token
    assert config.use_tactile_tcn is enabled
    assert config.vlm_tactile_reads_state is not enabled
    assert config.vlm_state_reads_tactile is enabled
    assert config.vlm_context_reads_tactile is not enabled
    assert config.fm_tactile_reads_vlm_tactile is not enabled
    config.save_pretrained(tmp_path / "new")
    serialized = json.loads((tmp_path / "new/config.json").read_text())
    for old, (new, _) in TACTILE_CONFIG_ALIASES.items():
        assert old not in serialized and new in serialized


def test_explicit_new_fields_and_cli_override_legacy_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "smolvla",
                "device": "cpu",
                "use_tactile": True,
                "use_vlm_tactile_token": True,
                "use_tactile_token": True,
                "use_fm_tactile_token": False,
                "vlm_tactile_mask_state": True,
                "vlm_tactile_reads_state": True,
            }
        )
    )
    config = SmolVLAConfig.from_pretrained(tmp_path)
    assert not config.use_fm_tactile_token and config.vlm_tactile_reads_state
    config = SmolVLAConfig.from_pretrained(
        tmp_path,
        cli_overrides=[
            "--use_tactile_token=true",
            "--vlm_tactile_mask_state=true",
            "--vlm_no_tactile_feedback=false",
            "--fm_tactile_no_vlm_tactile_attention=true",
        ],
    )
    assert config.use_fm_tactile_token and not config.vlm_tactile_reads_state
    assert config.vlm_context_reads_tactile and not config.fm_tactile_reads_vlm_tactile
    config = SmolVLAConfig.from_pretrained(tmp_path, cli_overrides=["--vlm_tactile_reads_state=false"])
    assert not config.vlm_tactile_reads_state


def test_default_semantics_and_alias_validation():
    config = SmolVLAConfig()
    assert not config.use_fm_tactile_token
    assert not config.use_tactile_tcn
    assert not config.vlm_tactile_reads_state and not config.vlm_state_reads_tactile
    assert config.vlm_context_reads_tactile and config.fm_tactile_reads_vlm_tactile
    source = {"type": "act", "use_tactile_token": True}
    assert migrate_tactile_policy_fields(source) is source
    with pytest.raises(ValueError, match="must be a boolean"):
        migrate_tactile_policy_fields({"vlm_tactile_mask_state": "false"})
    assert migrate_tactile_cli_args(
        [
            "--policy.use_tactile_token",
            "true",
            "--policy.use_tactile_token_history=true",
            "--policy.vlm_no_tactile_feedback=false",
            "--steps=5",
        ],
        prefix="policy.",
    ) == [
        "--policy.use_fm_tactile_token=true",
        "--policy.use_tactile_tcn=true",
        "--policy.vlm_context_reads_tactile=true",
        "--steps=5",
    ]


def test_layered_legacy_overrides_new_defaults_before_merge(tmp_path):
    (tmp_path / "defaults.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "repo_id": "test/data",
                "policy": {
                    "type": "smolvla",
                    "use_fm_tactile_token": False,
                    "vlm_tactile_reads_state": False,
                    "vlm_context_reads_tactile": True,
                    "fm_tactile_reads_vlm_tactile": True,
                },
            }
        )
    )
    child = tmp_path / "20261122_001.yaml"
    child.write_text(
        yaml.safe_dump(
            {
                "extends": "defaults.yaml",
                "policy": {
                    "use_tactile": True,
                    "use_vlm_tactile_token": True,
                    "use_tactile_token": True,
                    "vlm_tactile_mask_state": False,
                    "vlm_no_tactile_feedback": True,
                    "fm_tactile_no_vlm_tactile_attention": True,
                },
            }
        )
    )
    resolved = resolve_experiment_config(child)
    policy = resolved.config["policy"]
    assert policy["use_fm_tactile_token"] and policy["vlm_tactile_reads_state"]
    assert not policy["vlm_context_reads_tactile"] and not policy["fm_tactile_reads_vlm_tactile"]
    assert not set(TACTILE_CONFIG_ALIASES).intersection(policy)
    grandchild = tmp_path / "20261122_002.yaml"
    grandchild.write_text(
        yaml.safe_dump({"extends": child.name, "policy": {"vlm_context_reads_tactile": True}})
    )
    assert resolve_experiment_config(grandchild).config["policy"]["vlm_context_reads_tactile"]


def test_plain_yaml_path_overrides_and_training_resume_migration(tmp_path):
    raw = {
        "policy": {
            "type": "smolvla",
            "path": "test/base",
            "use_tactile_token": True,
            "vlm_no_tactile_feedback": True,
        }
    }
    path = tmp_path / "plain.yaml"
    path.write_text(yaml.safe_dump(raw))
    converted = parser.extract_path_fields_from_config(str(path), ["policy"])
    assert converted != str(path)
    overrides = parser.get_yaml_overrides("policy")
    assert "--use_fm_tactile_token=true" in overrides
    assert "--vlm_context_reads_tactile=false" in overrides
    checkpoint = tmp_path / "train_config.json"
    checkpoint.write_text(
        json.dumps(
            {
                "dataset": {"repo_id": "test/data"},
                "policy": {
                    "type": "smolvla",
                    "device": "cpu",
                    "use_tactile": True,
                    "use_tactile_token": True,
                    "vlm_tactile_mask_state": True,
                },
            }
        )
    )
    config = TrainPipelineConfig.from_pretrained(checkpoint)
    assert config.policy.use_fm_tactile_token and not config.policy.vlm_tactile_reads_state
    overridden = TrainPipelineConfig.from_pretrained(
        checkpoint, cli_args=["--policy.use_tactile_token=false", "--policy.vlm_tactile_mask_state=false"]
    )
    assert not overridden.policy.use_fm_tactile_token and overridden.policy.vlm_tactile_reads_state

"""Migrate serialized SmolVLA tactile options without changing their behavior."""

from typing import Any

TACTILE_CONFIG_ALIASES = {
    "use_tactile_token": ("use_fm_tactile_token", False),
    "use_tactile_token_history": ("use_tactile_tcn", False),
    "vlm_tactile_mask_state": ("vlm_tactile_reads_state", True),
    "state_reads_vlm_tactile": ("vlm_state_reads_tactile", False),
    "vlm_no_tactile_feedback": ("vlm_context_reads_tactile", True),
    "fm_tactile_no_vlm_tactile_attention": ("fm_tactile_reads_vlm_tactile", True),
}


def migrate_tactile_policy_fields(config: dict[str, Any]) -> dict[str, Any]:
    """Accept old names, prefer explicit new names, and remove aliases before parsing."""
    if config.get("type", "smolvla") != "smolvla":
        return config
    result = dict(config)
    result.pop("tactile_tcn_width", None)  # TCN now uses the action expert width throughout.
    for old, (new, invert) in TACTILE_CONFIG_ALIASES.items():
        if old not in result:
            continue
        value = result.pop(old)
        if not isinstance(value, bool):
            raise ValueError(f"Legacy tactile option {old} must be a boolean")
        result.setdefault(new, not value if invert else value)
    return result


def migrate_tactile_pipeline_fields(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize a nested training/evaluation policy mapping, preserving other fields."""
    policy = config.get("policy")
    if not isinstance(policy, dict):
        return config
    migrated = migrate_tactile_policy_fields(policy)
    if migrated == policy:
        return config
    return {**config, "policy": migrated}


def migrate_tactile_cli_args(args: list[str], *, prefix: str = "") -> list[str]:
    """Accept legacy boolean overrides in either --key=value or --key value form."""
    result = []
    index = 0
    while index < len(args):
        argument = args[index]
        key, separator, value = argument.partition("=")
        alias = key.removeprefix(f"--{prefix}") if key.startswith(f"--{prefix}") else ""
        if alias in TACTILE_CONFIG_ALIASES:
            new, invert = TACTILE_CONFIG_ALIASES[alias]
            if not separator:
                index += 1
                if index >= len(args):
                    raise ValueError(f"Missing value for {key}")
                value = args[index]
            if value.lower() not in ("true", "false"):
                raise ValueError(f"Legacy tactile option {alias} must be a boolean")
            enabled = value.lower() == "true"
            result.append(f"--{prefix}{new}={str(not enabled if invert else enabled).lower()}")
        else:
            result.append(argument)
        index += 1
    return result

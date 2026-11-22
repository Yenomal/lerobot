"""Tactile history is loaded independently of camera/state observation delay."""

from types import SimpleNamespace

import pytest

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

pytestmark = pytest.mark.dataset


@pytest.mark.parametrize("tcn", [False, True])
def test_tactile_delta_indices_match_configured_length_in_both_modes(tcn):
    config = SmolVLAConfig(
        use_tactile=True,
        use_fm_tactile_token=True,
        use_tactile_tcn=tcn,
        tactile_token_history_length=10,
        n_obs_steps=1,
    )
    metadata = SimpleNamespace(
        fps=30,
        features={
            "observation.state": {},
            "observation.images.head": {},
            "observation.markers.left": {},
            "observation.tactile.right_marker": {},
            "action": {},
        },
    )
    timestamps = resolve_delta_timestamps(
        config, metadata, {"observation.markers.left": "observation.tactile.left_marker"}
    )
    assert timestamps["observation.state"] == timestamps["observation.images.head"] == [0.0]
    expected = [i / 30 for i in range(-9, 1)]
    assert (
        timestamps["observation.markers.left"] == timestamps["observation.tactile.right_marker"] == expected
    )


@pytest.mark.parametrize("fm,vlm", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("fm_length,vlm_length", [(2, 7), (12, 2)])
@pytest.mark.parametrize("delay_unit", ["tokens", "chunks"])
def test_separate_history_window_covers_enabled_branches_and_vlm_delay(
    fm, vlm, fm_length, vlm_length, delay_unit
):
    config = SmolVLAConfig(
        use_tactile=True,
        use_fm_tactile_token=fm,
        use_vlm_tactile_token=vlm,
        fm_tactile_token_history_length=fm_length,
        vlm_tactile_token_history_length=vlm_length,
        streaming=True,
        observation_delay_max_tokens=3 if delay_unit == "tokens" else 0,
        observation_delay_max_chunks=2 if delay_unit == "chunks" else 0,
    )
    delay = 3 if delay_unit == "tokens" else 2 * config.streaming_chunk_size
    count = max(fm_length if fm else 1, vlm_length + delay if vlm else 1)
    metadata = SimpleNamespace(
        fps=30, features={"observation.tactile.left_marker": {}, "observation.state": {}}
    )
    timestamps = resolve_delta_timestamps(config, metadata)
    assert timestamps["observation.tactile.left_marker"] == [i / 30 for i in range(1 - count, 1)]
    assert timestamps["observation.state"] == [0.0]

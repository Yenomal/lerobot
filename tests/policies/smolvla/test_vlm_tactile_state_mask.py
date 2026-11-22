"""VLM tactile state visibility is independent of tactile feedback controls."""

import json
from dataclasses import replace

import pytest
import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
from tests.policies.smolvla import test_delay_adarms

make_model = test_delay_adarms.make_model


@pytest.mark.parametrize("mask_state", [False, True])
@pytest.mark.parametrize("no_vlm", [False, True])
@pytest.mark.parametrize("feedback", [False, True])
@pytest.mark.parametrize("tcn", [False, True])
@pytest.mark.parametrize("state_reads", [False, True])
def test_state_mask_prefix_invariance_and_training_cache_parity(
    make_model, monkeypatch, mask_state, no_vlm, feedback, tcn, state_reads
):
    torch.manual_seed(59)
    template = make_model(enabled=False, tactile=True)
    model = VLAFlowMatching(
        replace(
            template.config,
            use_vlm_tactile_token=True,
            vlm_tactile_reads_state=not (mask_state),
            vlm_state_reads_tactile=state_reads,
            tactile_no_vlm_attention=no_vlm,
            vlm_context_reads_tactile=not (feedback),
            use_tactile_token_history=tcn,
            tactile_token_history_length=3,
            prefix_length=12,
        )
    )
    batch = {f"observation.tactile.{side}_marker": torch.randn(2, 3, 2, 3, 2) for side in ("left", "right")}
    lang, valid = torch.tensor([[1, 2], [3, 4]]), torch.tensor([[True, False], [True, True]])
    state = torch.randn(2, 6)
    captured = []
    original = model.vlm_with_expert.forward

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append((kwargs["attention_mask"].clone(), result[0][0]))
        return result

    monkeypatch.setattr(model.vlm_with_expert, "forward", capture)
    cache = model.encode_prefix([], [], lang, valid, state, tactile_batch=batch)
    mask, hidden = captured[-1]
    count = 1 if tcn else 3
    expected_visible = not (mask_state or no_vlm)
    assert mask[:, 3 : 3 + count, 2].all().item() == expected_visible
    assert not mask[:, :2, 2].any()  # Image/language -> state unchanged.
    assert mask[:, 2, 3 : 3 + count].all().item() == (state_reads and not feedback)
    assert not mask[:, :, 3 + count :].any()  # Prefix padding is still hidden.
    model.encode_prefix([], [], lang, valid, -10 * state, tactile_batch=batch)
    changed = captured[-1][1]
    if expected_visible:
        assert not torch.allclose(hidden[:, 3 : 3 + count], changed[:, 3 : 3 + count])
    else:
        torch.testing.assert_close(hidden[:, 3 : 3 + count], changed[:, 3 : 3 + count], rtol=0, atol=0)
    actions, noise, time = torch.randn(2, 4, 6), torch.randn(2, 4, 6), torch.full((2, 4), 0.5)
    raw, copy = model.denoise_step(
        cache["prefix_pad_masks"],
        cache["past_key_values"],
        0.5 * (actions + noise),
        time,
        tactile_batch=batch,
        return_tactile_copy=True,
        apply_head_guidance=False,
    )
    losses = model([], [], lang, valid, state, actions, noise, time, tactile_batch=batch)
    expected = (raw - (noise - actions)).square()
    expected[:, 0] = 0.5 * (expected[:, 0] + (copy - (noise - actions)[:, 0]).square())
    torch.testing.assert_close(losses, expected)
    torch.testing.assert_close(captured[-1][0][:, :12, :12], mask)


def test_state_mask_default_legacy_loading_and_explicit_false(tmp_path):
    assert not (SmolVLAConfig().vlm_tactile_reads_state)
    config = SmolVLAConfig(use_tactile=True, use_vlm_tactile_token=True, vlm_tactile_reads_state=True)
    config.save_pretrained(tmp_path)
    assert SmolVLAConfig.from_pretrained(tmp_path).vlm_tactile_reads_state
    path = tmp_path / "config.json"
    old = json.loads(path.read_text())
    del old["vlm_tactile_reads_state"]
    path.write_text(json.dumps(old))
    assert not (SmolVLAConfig.from_pretrained(tmp_path).vlm_tactile_reads_state)


def test_reverse_state_visibility_default_and_legacy_loading(tmp_path):
    assert not SmolVLAConfig().vlm_state_reads_tactile
    config = SmolVLAConfig(use_tactile=True, use_vlm_tactile_token=True, vlm_state_reads_tactile=True)
    config.save_pretrained(tmp_path)
    assert SmolVLAConfig.from_pretrained(tmp_path).vlm_state_reads_tactile
    path = tmp_path / "config.json"
    old = json.loads(path.read_text())
    del old["vlm_state_reads_tactile"]
    path.write_text(json.dumps(old))
    assert not SmolVLAConfig.from_pretrained(tmp_path).vlm_state_reads_tactile

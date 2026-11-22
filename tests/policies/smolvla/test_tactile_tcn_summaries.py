"""Five-frame, full-width causal temporal summaries preserve every input block."""

from dataclasses import fields, replace

import pytest
import torch

from lerobot.configs.smolvla_compat import migrate_tactile_policy_fields
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
from tests.policies.smolvla import test_delay_adarms

make_model = test_delay_adarms.make_model


@pytest.mark.parametrize("length", [5, 10, 20])
@pytest.mark.parametrize("fingers", [False, True])
def test_tcn_local_receptive_fields_widths_and_finger_batching(make_model, length, fingers):
    torch.manual_seed(71)
    model = VLAFlowMatching(
        replace(
            make_model(tactile=True, enabled=False).config,
            use_vlm_tactile_token=True,
            use_tactile_tcn=True,
            tactile_token_history_length=length,
            use_tactile_finger_tokens=fingers,
        )
    )
    data = {
        f"observation.tactile.{side}_marker": torch.randn(2, length, 2, 3, 2) for side in ("left", "right")
    }
    multiplier = 2 if fingers else 1
    for vlm, prefix in ((False, ""), (True, "vlm_")):
        tokens = model._encode_tactile(data, vlm=vlm)
        assert tokens.shape == (2, length // 5 * multiplier, 8)
        for name in ("tactile_token_tcn_in", "tactile_token_out"):
            layer = getattr(model, prefix + name)
            assert layer.in_features == layer.out_features == 8
        for name in ("tactile_token_tcn_1", "tactile_token_tcn_2"):
            layer = getattr(model, prefix + name)
            assert (layer.in_features, layer.out_features) == (24, 8)
        for frame in range(length):
            changed = {key: value.clone() for key, value in data.items()}
            changed["observation.tactile.left_marker"][:, frame] *= -10
            actual = model._encode_tactile(changed, vlm=vlm)
            keep = torch.ones(tokens.shape[1], dtype=torch.bool)
            begin = frame // 5 * multiplier
            keep[begin : begin + (1 if fingers else multiplier)] = False
            torch.testing.assert_close(tokens[:, keep], actual[:, keep], rtol=0, atol=0)
            assert not torch.allclose(tokens[:, begin], actual[:, begin])


@pytest.mark.parametrize("fingers", [False, True])
@pytest.mark.parametrize("amp", [False, True])
def test_multisummary_rope_conditions_and_training_cached_parity(make_model, monkeypatch, fingers, amp):
    torch.manual_seed(72)
    model = VLAFlowMatching(
        replace(
            make_model(tactile=True, enabled=False, tactile_cross_attention=True).config,
            use_vlm_tactile_token=True,
            use_tactile_tcn=True,
            vlm_tactile_token_history_length=20,
            fm_tactile_token_history_length=10,
            use_tactile_finger_tokens=fingers,
            prefix_length=16,
        )
    )
    data = {f"observation.tactile.{side}_marker": torch.randn(2, 20, 2, 3, 2) for side in ("left", "right")}
    lang, valid, state = torch.tensor([[1, 2], [3, 4]]), torch.ones(2, 2, dtype=torch.bool), torch.randn(2, 6)
    actions, noise, time = torch.randn(2, 4, 6), torch.randn(2, 4, 6), torch.full((2, 4), 0.5)
    captured = []
    original = model.vlm_with_expert.forward

    def capture(*args, **kwargs):
        captured.append((kwargs["position_ids"].clone(), kwargs["attention_mask"].clone()))
        return original(*args, **kwargs)

    monkeypatch.setattr(model.vlm_with_expert, "forward", capture)
    multiplier = 2 if fingers else 1
    fc, vc = 2 * multiplier, 4 * multiplier
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        cache = model.encode_prefix([], [], lang, valid, state, tactile_batch=data)
        expected_prefix = torch.tensor([7, 12, 17, 22]).repeat_interleave(multiplier)
        torch.testing.assert_close(captured[-1][0][:, 3 : 3 + vc], expected_prefix[None].expand(2, -1))
        roles = model.embed_prefix([], [], lang, valid, state, tactile_batch=data)[2]
        torch.testing.assert_close(model._prefix_tactile_keys(cache["prefix_pad_masks"]), roles == 2)
        raw, ca0 = model.denoise_step(
            cache["prefix_pad_masks"],
            cache["past_key_values"],
            0.5 * (actions + noise),
            time,
            tactile_batch=data,
            return_tactile_copy=True,
            apply_head_guidance=False,
        )
        cached_positions, cached_mask = captured[-1]
        expected_fm = torch.tensor([27, 32]).repeat_interleave(multiplier)
        torch.testing.assert_close(cached_positions[:, :fc], expected_fm[None].expand(2, -1))
        loss = model([], [], lang, valid, state, actions, noise, time, tactile_batch=data)
    suffix = fc + 1 + 4
    torch.testing.assert_close(captured[-1][0][:, -suffix:], cached_positions)
    torch.testing.assert_close(captured[-1][1][:, -suffix:], cached_mask)
    expected = (raw - (noise - actions)).square()
    expected[:, 0] = 0.5 * (expected[:, 0] + (ca0 - (noise - actions)[:, 0]).square())
    torch.testing.assert_close(loss, expected)
    loss.mean().backward()
    assert model.tactile_token_tcn_1.weight.grad.abs().sum() > 0
    assert model.vlm_tactile_token_tcn_1.weight.grad.abs().sum() > 0


def test_tcn_history_validation_and_removed_width():
    for fm, vlm in ((True, False), (False, True)):
        for length in (1, 4, 7, 19):
            with pytest.raises(ValueError, match="multiple of 5"):
                SmolVLAConfig(
                    use_tactile=True,
                    use_fm_tactile_token=fm,
                    use_vlm_tactile_token=vlm,
                    use_tactile_tcn=True,
                    tactile_token_history_length=length,
                )
    assert "tactile_tcn_width" not in {field.name for field in fields(SmolVLAConfig)}
    assert "tactile_tcn_width" not in migrate_tactile_policy_fields(
        {"type": "smolvla", "tactile_tcn_width": 256}
    )

"""Independent controls for VLM tactile feedback and FM tactile prefix reads."""

from dataclasses import replace

import pytest
import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
from tests.policies.smolvla import test_delay_adarms

make_model = test_delay_adarms.make_model


@pytest.mark.parametrize("attention_mode", ["self_attn", "cross_attn"])
@pytest.mark.parametrize("no_feedback", [False, True])
@pytest.mark.parametrize("no_fm_read", [False, True])
@pytest.mark.parametrize("tcn", [False, True])
@pytest.mark.parametrize("amp", [False, True])
def test_flags_masks_cache_parity_and_hidden_invariance(
    make_model, monkeypatch, attention_mode, no_feedback, no_fm_read, tcn, amp
):
    torch.manual_seed(52)
    template = make_model(
        enabled=False,
        tactile=True,
        attention_mode=attention_mode,
        tactile_cross_attention=attention_mode == "cross_attn",
    )
    model = VLAFlowMatching(
        replace(
            template.config,
            use_vlm_tactile_token=True,
            use_tactile_ca0=False,
            vlm_context_reads_tactile=not (no_feedback),
            fm_tactile_reads_vlm_tactile=not (no_fm_read),
            use_tactile_token_history=tcn,
            fm_tactile_token_history_length=3,
            vlm_tactile_token_history_length=2,
            prefix_length=12,
        )
    )
    for layer in model.vlm_with_expert.lm_expert.layers:
        torch.nn.init.normal_(layer.self_attn.o_proj.weight, std=0.5)
    batch = {f"observation.tactile.{side}_marker": torch.randn(2, 3, 2, 3, 2) for side in ("left", "right")}
    lang, valid, state = torch.tensor([[1, 2], [3, 4]]), torch.ones(2, 2, dtype=torch.bool), torch.randn(2, 6)
    calls = []
    original_forward = model.vlm_with_expert.forward

    def capture(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        calls.append((kwargs["attention_mask"].clone(), result[0]))
        return result

    monkeypatch.setattr(model.vlm_with_expert, "forward", capture)
    actions, noise, time = torch.randn(2, 4, 6), torch.randn(2, 4, 6), torch.full((2, 4), 0.5)

    def run():
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            cache = model.encode_prefix([], [], lang, valid, state, tactile_batch=batch)
            prefix_mask, prefix_hidden = calls[-1][0], calls[-1][1][0]
            velocity = model.denoise_step(
                cache["prefix_pad_masks"],
                cache["past_key_values"],
                0.5 * (actions + noise),
                time,
                tactile_batch=batch,
                apply_head_guidance=False,
            )
        fm_mask, suffix_hidden = calls[-1][0], calls[-1][1][1]
        return prefix_mask, prefix_hidden, fm_mask, suffix_hidden, velocity

    prefix_mask, prefix_hidden, fm_mask, suffix_hidden, velocity = run()
    vlm_count, fm_count = (1, 1) if tcn else (2, 3)
    t_end = 3 + vlm_count
    assert prefix_mask[:, 3:t_end, :2].all()  # T_V continues to read language.
    if no_feedback:
        assert not prefix_mask[:, :3, 3:t_end].any()
    else:
        assert prefix_mask[:, :2, 3:t_end].all()
    assert not prefix_mask[:, 3:t_end, 2].any()  # State remains hidden from T_V.
    assert fm_mask[:, :fm_count, :3].all()  # T_F retains image/language/state context.
    if no_fm_read:
        assert not fm_mask[:, :fm_count, 3:t_end].any()
    else:
        assert fm_mask[:, :fm_count, 3:t_end].all()
    assert not fm_mask[:, fm_count : fm_count + 2, 3:t_end].any()  # Existing first-chunk rule.
    assert fm_mask[:, fm_count + 2 :, 3:t_end].all()  # Later actions stay unchanged.
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        loss = model([], [], lang, valid, state, actions, noise, time, tactile_batch=batch)
    torch.testing.assert_close(loss, (velocity - (noise - actions)).square())
    torch.testing.assert_close(calls[-1][0][:, -fm_mask.shape[1] :], fm_mask)

    # Perturb only the independent VLM encoder path, keeping FM tactile inputs fixed.
    original_encode = model._encode_tactile

    def changed_encode(batch, *, vlm=False):
        result = original_encode(batch, vlm=vlm)
        return -7 * result + 0.3 if vlm else result

    monkeypatch.setattr(model, "_encode_tactile", changed_encode)
    _, changed_prefix, _, changed_suffix, _ = run()
    if no_feedback:
        torch.testing.assert_close(prefix_hidden[:, :3], changed_prefix[:, :3], rtol=0, atol=0)
    else:
        assert not torch.allclose(prefix_hidden[:, :2], changed_prefix[:, :2])
    if no_feedback and no_fm_read:
        torch.testing.assert_close(suffix_hidden[:, :fm_count], changed_suffix[:, :fm_count], rtol=0, atol=0)
    else:
        assert not torch.allclose(suffix_hidden[:, :fm_count], changed_suffix[:, :fm_count])


def test_circuit_flags_round_trip_and_validation(tmp_path):
    config = SmolVLAConfig(
        use_tactile=True,
        use_fm_tactile_token=True,
        use_vlm_tactile_token=True,
        vlm_context_reads_tactile=False,
        fm_tactile_reads_vlm_tactile=False,
    )
    config.save_pretrained(tmp_path)
    loaded = SmolVLAConfig.from_pretrained(tmp_path)
    assert not (loaded.vlm_context_reads_tactile) and not (loaded.fm_tactile_reads_vlm_tactile)
    assert SmolVLAConfig().vlm_context_reads_tactile
    assert SmolVLAConfig().fm_tactile_reads_vlm_tactile
    with pytest.raises(ValueError, match="vlm_context_reads_tactile requires"):
        SmolVLAConfig(vlm_context_reads_tactile=False)
    for fm, vlm in ((False, True), (True, False)):
        with pytest.raises(ValueError, match="requires both VLM and FM"):
            SmolVLAConfig(
                use_tactile=True,
                use_fm_tactile_token=fm,
                use_vlm_tactile_token=vlm,
                fm_tactile_reads_vlm_tactile=False,
            )

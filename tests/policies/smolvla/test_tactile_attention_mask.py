"""CA0 copies A0's action context while keeping tactile out of real actions."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
from tests.policies.smolvla import test_delay_adarms

make_model = test_delay_adarms.make_model


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("mode", ["bidirectional", "mask", "causal", "random_gate", "nonstream"])
@pytest.mark.parametrize("delay_count", [0, 1, 4])
@pytest.mark.parametrize("tactile_count", [1, 3])
@pytest.mark.parametrize("tactile_causal", [False, True])
@pytest.mark.parametrize("read_prefix", [False, True])
def test_ca0_inherits_a0_context_and_real_actions_preserve_visibility(
    cached, mode, delay_count, tactile_count, tactile_causal, read_prefix
):
    prefix_len, horizon = 3, 4
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        chunk_size=horizon,
        streaming=mode != "nonstream",
        streaming_chunk_size=2,
        streaming_attention_mode=mode,
    )
    tactile = prefix_len + delay_count
    copy, action0 = tactile + tactile_count, tactile + tactile_count + 1
    length = action0 + horizon
    valid = torch.ones(2, length, dtype=torch.bool)
    valid[0, 1] = False  # Padded prefix key.
    valid[1, -1] = False  # A padded action must not become visible to CA0.
    boundaries = torch.zeros_like(valid)
    if delay_count:
        boundaries[:, prefix_len] = True
    boundaries[:, copy] = True
    boundaries[:, action0:] = model._action_att_masks(2, "cpu", torch.bool)
    offset = prefix_len if cached else 0
    original = make_att_2d_masks(valid, boundaries)[:, offset:]
    if delay_count:
        original = model._apply_delay_token_mask(
            original,
            prefix_len=prefix_len,
            action_count=horizon,
            query_offset=offset,
            action_start_offset=delay_count + tactile_count + 1,
            delay_token_count=delay_count,
        )
    result = model._apply_tactile_token_mask(
        original,
        prefix_len=prefix_len,
        action_count=horizon,
        query_offset=offset,
        delay_token=delay_count,
        tactile_token_count=tactile_count,
        causal=tactile_causal,
        read_prefix=read_prefix,
    )
    # A0 and later actions preserve all real-action edges, including reading A0.
    torch.testing.assert_close(
        result[:, action0 - offset :, action0:], original[:, action0 - offset :, action0:]
    )
    torch.testing.assert_close(result[:, action0 - offset :, action0], valid[:, action0:])
    assert not result[:, action0 - offset :, tactile : copy + 1].any()
    # CA0 sees exactly the same other actions and prefix/delay as A0.
    ca0_row, a0_row = result[:, copy - offset], result[:, action0 - offset]
    torch.testing.assert_close(ca0_row[:, action0 + 1 :], a0_row[:, action0 + 1 :])
    torch.testing.assert_close(ca0_row[:, :tactile], a0_row[:, :tactile])
    assert ca0_row[:, tactile:copy].all()
    expected_tactile = torch.ones(2, tactile_count, tactile_count, dtype=torch.bool)
    if tactile_causal:
        expected_tactile = expected_tactile.tril()
    torch.testing.assert_close(result[:, tactile - offset : copy - offset, tactile:copy], expected_tactile)
    tactile_prefix = result[:, tactile - offset : copy - offset, :prefix_len]
    if read_prefix:
        torch.testing.assert_close(tactile_prefix, valid[:, None, :prefix_len].expand(-1, tactile_count, -1))
    else:
        assert not tactile_prefix.any()
    assert ca0_row[:, copy].all()
    assert not ca0_row[:, action0].any()
    assert a0_row[:, action0].all()
    assert not a0_row[:, tactile : copy + 1].any()
    # This explicitly exercises CA0's repaired forward edges in bidirectional
    # modes and its inherited causal chunk restriction in the other modes.
    if mode in ("bidirectional", "mask", "random_gate"):
        assert ca0_row[0, action0 + 1 :].all()
    elif mode == "causal":
        assert ca0_row[0, action0 + 1]
        assert not ca0_row[0, action0 + 2 :].any()
    else:
        assert not ca0_row[:, action0 + 1 :].any()
    assert not result[:, tactile - offset : copy - offset, prefix_len:tactile].any()
    assert not result[:, tactile - offset : copy - offset, copy:].any()
    assert not result[0, :, 1].any()
    assert not ca0_row[1, -1]
    if not cached:
        assert not result[:, :prefix_len, tactile:].any()


@pytest.mark.parametrize("cross_tactile", [False, True])
def test_a0_and_ca0_receive_other_action_context_without_tactile_leakage(make_model, cross_tactile):
    torch.manual_seed(39)
    model = make_model(enabled=False, tactile=True, tactile_cross_attention=cross_tactile)
    # Exercise actual self-attention followed by cross-attention, with real K/V caches.
    for layer in model.vlm_with_expert.lm_expert.layers:
        nn.init.normal_(layer.self_attn.o_proj.weight, std=0.5)
    lang = torch.tensor([[1, 2]])
    cache = model.encode_prefix([], [], lang, torch.ones_like(lang, dtype=torch.bool), torch.randn(1, 6))
    tactile = torch.randn(1, 8)
    actions, time = torch.randn(1, 4, 6), torch.full((1, 4), 0.5)

    def predict(x, embedding):
        return model.denoise_step(
            cache["prefix_pad_masks"],
            cache["past_key_values"],
            x,
            time,
            tactile_embedding=embedding,
            apply_head_guidance=False,
            return_tactile_copy=True,
        )

    raw, copy = predict(actions, tactile)
    changed_actions = actions.clone()
    changed_actions[:, 1:] *= -10
    changed_raw, changed_copy = predict(changed_actions, tactile)
    assert not torch.allclose(raw[:, 0], changed_raw[:, 0])
    assert not torch.allclose(copy, changed_copy)
    tactile_raw, tactile_copy = predict(actions, -10 * tactile)
    torch.testing.assert_close(raw, tactile_raw, rtol=0, atol=0)
    assert not torch.allclose(copy, tactile_copy)

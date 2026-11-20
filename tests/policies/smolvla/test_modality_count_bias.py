"""Visible-count correction matches hierarchical attention without changing VLM queries."""

import pytest
import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from tests.policies.smolvla import test_delay_adarms

make_model = test_delay_adarms.make_model


@pytest.mark.parametrize("fm_start", [0, 2])
def test_matches_two_stage_attention_and_gradients_with_query_specific_counts(fm_start):
    torch.manual_seed(7)
    # P P P P | T D | CA0 A0 A1. Different visibility for each query/sample.
    mask = torch.rand(2, 5, 9) > 0.3
    mask[:, 0, 4:6] = False  # Entire auxiliary modality absent.
    mask[:, 1, :4] = False  # Entire prefix modality absent.
    mask[:, 2, 4:] = False  # Only one modality remains.
    mask[:, :, -1] = False  # Padded key.
    scores = torch.randn(2, 3, 5, 9, requires_grad=True)
    existing_bias = torch.randn(2, 1, 5, 9) * 0.2
    logits = scores + existing_bias
    values = torch.randn(2, 3, 9, 4, requires_grad=True)
    bias = SmolVLMWithExpertModel._modality_count_bias(
        mask, prefix_key_count=4, auxiliary_key_count=2, fm_query_start=fm_start
    )
    assert torch.isfinite(bias).all()
    assert not bias[:, :fm_start].any()
    probs = (logits + bias[:, None]).masked_fill(~mask[:, None], -torch.inf).softmax(-1)
    actual = probs @ values
    expected_rows = []
    for b in range(2):
        heads = []
        for h in range(3):
            rows = []
            for q in range(5):
                if q < fm_start:
                    weights = logits[b, h, q].masked_fill(~mask[b, q], -torch.inf).softmax(-1)
                    rows.append(weights @ values[b, h])
                    continue
                group_scores, group_outputs = [], []
                for lo, hi in ((0, 4), (4, 6), (6, 9)):
                    visible = torch.arange(lo, hi)[mask[b, q, lo:hi]]
                    if visible.numel() == 0:
                        continue
                    s = logits[b, h, q, visible]
                    group_scores.append(s.logsumexp(0) - s.new_tensor(visible.numel()).log())
                    group_outputs.append(s.softmax(0) @ values[b, h, visible])
                weights = torch.stack(group_scores).softmax(0)
                rows.append((weights[:, None] * torch.stack(group_outputs)).sum(0))
            heads.append(torch.stack(rows))
        expected_rows.append(torch.stack(heads))
    expected = torch.stack(expected_rows)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().sum(), (scores, values), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), (scores, values))
    for a, e in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, e)


def test_equal_scores_equal_modality_mass_and_duplication_invariance():
    def attention(prefix_repeats):
        prefix = torch.tensor([1.0, 3.0]).repeat(prefix_repeats)
        values = torch.cat([prefix, torch.tensor([10.0, 20.0, 30.0])])
        mask = torch.ones(1, 1, values.numel(), dtype=torch.bool)
        bias = SmolVLMWithExpertModel._modality_count_bias(
            mask, prefix_key_count=prefix.numel(), auxiliary_key_count=1
        )
        probs = bias.softmax(-1)[0, 0]
        torch.testing.assert_close(probs[: prefix.numel()].sum(), torch.tensor(1 / 3))
        torch.testing.assert_close(probs[prefix.numel()], torch.tensor(1 / 3))
        torch.testing.assert_close(probs[-2:].sum(), torch.tensor(1 / 3))
        return probs @ values

    torch.testing.assert_close(attention(1), attention(100))


@pytest.mark.parametrize("mode", ["self_attn", "cross_attn"])
@pytest.mark.parametrize("delay", [False, True])
def test_no_tactile_training_cache_parity_and_unchanged_prefix(make_model, mode, delay):
    torch.manual_seed(10)
    model = make_model(mode, enabled=False, delay_token=delay, sincos=delay, modality_count_bias=True)
    lang = torch.tensor([[1, 2], [3, 4]])
    valid = torch.tensor([[True, False], [True, True]])
    state = torch.randn(2, 6)
    actions, noise = torch.randn(2, 4, 6), torch.randn(2, 4, 6)
    time = torch.tensor([[0.2, 0.2, 0.8, 0.8]]).expand(2, -1)
    x_t = time[..., None] * noise + (1 - time[..., None]) * actions
    expert = model.vlm_with_expert
    expert.use_modality_count_bias = False
    baseline_cache = model.encode_prefix([], [], lang, valid, state)
    expert.use_modality_count_bias = True
    cache = model.encode_prefix([], [], lang, valid, state)
    for old, new in zip(
        baseline_cache["past_key_values"].layers, cache["past_key_values"].layers, strict=True
    ):
        torch.testing.assert_close(old.keys, new.keys, rtol=0, atol=0)
        torch.testing.assert_close(old.values, new.values, rtol=0, atol=0)
    velocity = model.denoise_step(
        cache["prefix_pad_masks"], cache["past_key_values"], x_t, time, apply_head_guidance=False
    )
    loss = model([], [], lang, valid, state, actions, noise=noise, time=time)
    torch.testing.assert_close(loss, (noise - actions - velocity).square())
    loss.mean().backward()
    assert model.action_in_proj.weight.grad.abs().sum() > 0


def test_config_round_trip_and_legacy_default(tmp_path):
    assert not SmolVLAConfig(device="cpu").use_modality_count_bias
    config = SmolVLAConfig(device="cpu", use_modality_count_bias=True)
    config.save_pretrained(tmp_path)
    assert SmolVLAConfig.from_pretrained(tmp_path).use_modality_count_bias


def test_attention_with_no_visible_keys_returns_zero_and_no_value_gradient():
    model = object.__new__(SmolVLMWithExpertModel)
    torch.nn.Module.__init__(model)
    model.num_attention_heads = model.num_key_value_heads = 1
    query = torch.randn(1, 2, 1, 4)
    key = torch.randn(1, 3, 1, 4)
    value = torch.randn(1, 3, 1, 4, requires_grad=True)
    mask = torch.tensor([[[False, False, False], [True, True, True]]])
    output = model.eager_attention_forward(mask, 1, 4, query, key, value)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[:, 0], torch.zeros(1, 4), rtol=0, atol=0)
    empty_row_grad = torch.autograd.grad(output[:, 0].sum(), value)[0]
    assert torch.count_nonzero(empty_row_grad) == 0
    reference = (
        torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=mask[:, None]
        )
        .transpose(1, 2)
        .reshape(1, 2, 4)
    )
    torch.testing.assert_close(output, reference)

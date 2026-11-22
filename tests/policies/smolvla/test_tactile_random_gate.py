"""Action chunk gates coexist with independent tactile masks and optional CA0."""

from dataclasses import replace

import pytest
import torch

from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
from tests.policies.smolvla import test_delay_adarms

make_model = test_delay_adarms.make_model


@pytest.mark.parametrize("ca0", [False, True])
@pytest.mark.parametrize("vlm", [False, True])
@pytest.mark.parametrize("attention_mode", ["self_attn", "cross_attn"])
@pytest.mark.parametrize("tcn,causal", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("delay", [False, True])
@pytest.mark.parametrize("amp", [False, True])
def test_action_gates_preserve_tactile_masks_and_ca0_context(
    make_model, monkeypatch, ca0, vlm, attention_mode, tcn, causal, delay, amp
):
    torch.manual_seed(54)
    template = make_model(
        tactile=True,
        enabled=False,
        attention_mode=attention_mode,
        tactile_cross_attention=attention_mode == "cross_attn",
        delay_token=delay,
        sincos=delay,
        modality_count_bias=True,
    )
    model = VLAFlowMatching(
        replace(
            template.config,
            chunk_size=10,
            n_action_steps=10,
            streaming_chunk_size=5,
            streaming_attention_mode="random_gate",
            use_tactile_ca0=ca0,
            use_vlm_tactile_token=vlm,
            use_tactile_tcn=tcn,
            use_tactile_causal_attention=causal,
            fm_tactile_token_history_length=(5 if tcn else 3),
            vlm_tactile_token_history_length=(5 if tcn else 2),
            vlm_context_reads_tactile=not (vlm),
            fm_tactile_reads_vlm_tactile=not (vlm),
        )
    )
    for layer in model.vlm_with_expert.lm_expert.layers:
        torch.nn.init.normal_(layer.self_attn.o_proj.weight, std=0.5)
    batch = {f"observation.tactile.{side}_marker": torch.randn(2, 3, 2, 3, 2) for side in ("left", "right")}
    lang, valid, state = torch.tensor([[1, 2], [3, 4]]), torch.ones(2, 2, dtype=torch.bool), torch.randn(2, 6)
    actions, noise, time = torch.randn(2, 10, 6), torch.randn(2, 10, 6), torch.full((2, 10), 0.5)
    gates = torch.tensor([[0.25, 1.5], [0.5, 2.0]])
    monkeypatch.setattr(model, "_random_attention_log_gates", lambda *args: gates.log())
    captures = []
    forward = model.vlm_with_expert.forward

    def capture(*args, **kwargs):
        captures.append((kwargs["attention_mask"].clone(), kwargs.get("attention_bias")))
        return forward(*args, **kwargs)

    monkeypatch.setattr(model.vlm_with_expert, "forward", capture)

    def loss():
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            return model([], [], lang, valid, state, actions, noise, time, tactile_batch=batch)

    model.train()
    trained_loss = loss()
    mask, bias = captures[-1]
    assert bias is not None
    action_start = bias.shape[-1] - 10
    action_bias = bias[:, action_start:, action_start:]
    assert not action_bias[:, :5, :5].any() and not action_bias[:, 5:, 5:].any()
    torch.testing.assert_close(action_bias[:, :5, 5:], gates[:, :1, None].log().expand(-1, 5, 5))
    torch.testing.assert_close(action_bias[:, 5:, :5], gates[:, 1:, None].log().expand(-1, 5, 5))
    # With equal action logits the chunk mass ratio is exactly the sampled gate.
    probabilities = action_bias.softmax(-1)
    torch.testing.assert_close(probabilities[:, 0, 5:].sum(-1), gates[:, 0] / (1 + gates[:, 0]))
    allowed = torch.zeros_like(bias, dtype=torch.bool)
    allowed[:, action_start:, action_start:] = True
    if ca0:
        copy = action_start - 1
        allowed[:, copy, action_start:] = True
        torch.testing.assert_close(
            bias[:, copy, action_start + 1 :], bias[:, action_start, action_start + 1 :]
        )
        assert not bias[:, copy, copy].any() and not bias[:, copy, action_start].any()
    assert not bias.masked_select(~allowed).any()  # No prefix/tactile/delay gate.

    monkeypatch.setattr(model, "_random_attention_log_gates", lambda *args: torch.zeros_like(gates))
    ungated_loss = loss()
    torch.testing.assert_close(captures[-1][0], mask)
    assert not captures[-1][1].any()
    assert not torch.allclose(trained_loss, ungated_loss)
    trained_loss.mean().backward()
    assert model.tactile_encoder[0].weight.grad.abs().sum() > 0

    model.eval()
    eval_loss = loss()
    assert captures[-1][1] is None
    torch.testing.assert_close(eval_loss, ungated_loss)
    torch.testing.assert_close(captures[-1][0], mask)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        cache = model.encode_prefix([], [], lang, valid, state, tactile_batch=batch)
        result = model.denoise_step(
            cache["prefix_pad_masks"],
            cache["past_key_values"],
            0.5 * (actions + noise),
            time,
            tactile_batch=batch,
            return_tactile_copy=ca0,
            apply_head_guidance=False,
        )
    raw, copy = result if ca0 else (result, None)
    expected = (raw - (noise - actions)).square()
    if ca0:
        expected[:, 0] = 0.5 * (expected[:, 0] + (copy - (noise - actions)[:, 0]).square())
    torch.testing.assert_close(eval_loss, expected)
    assert captures[-1][1] is None

"""Optional CA0 versus direct first-chunk tactile routing."""

import json
from dataclasses import replace

import pytest
import torch

from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
from lerobot.policies.smolvla.smolvlm_with_expert import FlowTimeRMSNorm
from tests.policies.smolvla import test_delay_adarms

make_model = test_delay_adarms.make_model


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("delay", [0, 1, 10])
@pytest.mark.parametrize("all_actions", [False, True])
def test_action_fm_tactile_scope_preserves_vlm_prefix_mask(cached, causal, delay, all_actions):
    prefix, tactile, horizon = 8, 3, 10
    length = prefix + delay + tactile + horizon
    pad = torch.ones(2, length, dtype=torch.bool)
    pad[0, 1] = False
    boundaries = torch.zeros_like(pad)
    boundaries[:, prefix + delay] = True
    boundaries[:, prefix + delay + tactile] = True
    offset = prefix if cached else 0
    original = make_att_2d_masks(pad, boundaries)[:, offset:]
    vlm_keys = torch.zeros(2, prefix, dtype=torch.bool)
    vlm_keys[:, 5:] = True
    actual = VLAFlowMatching._apply_tactile_token_mask(
        original,
        prefix_len=prefix,
        action_count=horizon,
        query_offset=offset,
        delay_token=delay,
        tactile_token_count=tactile,
        causal=causal,
        use_ca0=False,
        tactile_action_count=5,
        prefix_tactile_mask=vlm_keys,
        all_actions=all_actions,
    )
    t_start, action = prefix + delay, prefix + delay + tactile
    q_action = action - offset
    assert actual[:, q_action : q_action + 5, t_start:action].all()
    assert (actual[:, q_action + 5 :, t_start:action] == all_actions).all()
    assert not actual[:, q_action : q_action + 5, 5:prefix].any()
    torch.testing.assert_close(actual[:, q_action + 5 :, :prefix], original[:, q_action + 5 :, :prefix])
    torch.testing.assert_close(actual[:, q_action:, action:], original[:, q_action:, action:])
    assert not actual[:, t_start - offset : q_action, action:].any()
    assert not actual[0, :, 1].any()


@pytest.mark.parametrize("ca0", [False, True])
@pytest.mark.parametrize("vlm", [False, True])
@pytest.mark.parametrize("attention_mode", ["self_attn", "cross_attn"])
@pytest.mark.parametrize("delay_route", ["none", "token", "sincos"])
@pytest.mark.parametrize("tcn", [False, True])
def test_layout_conditions_masks_loss_and_cached_parity(
    make_model, monkeypatch, ca0, vlm, attention_mode, delay_route, tcn
):
    torch.manual_seed(49)
    template = make_model(
        tactile=True,
        attention_mode=attention_mode,
        tactile_cross_attention=attention_mode == "cross_attn",
        enabled=True,
        use_noise_adarms=True,
        use_noise_embedding=False,
        delay_token=delay_route != "none",
        sincos=delay_route == "sincos",
    )
    model = VLAFlowMatching(
        replace(
            template.config,
            use_tactile_ca0=ca0,
            use_vlm_tactile_token=vlm,
            fm_tactile_token_history_length=(5 if tcn else 3),
            vlm_tactile_token_history_length=(5 if tcn else 2),
            use_tactile_tcn=tcn,
            prefix_length=12,
        )
    )
    for module in model.modules():
        if isinstance(module, FlowTimeRMSNorm):
            torch.nn.init.normal_(module.projection.weight, std=0.2)
    batch = {f"observation.tactile.{side}_marker": torch.randn(2, 3, 2, 3, 2) for side in ("left", "right")}
    lang, state = torch.tensor([[1, 2], [3, 4]]), torch.randn(2, 6)
    lang_mask = torch.tensor([[True, False], [True, True]])
    cache = model.encode_prefix([], [], lang, lang_mask, state, tactile_batch=batch)
    roles = model.embed_prefix([], [], lang, lang_mask, state, tactile_batch=batch)[2]
    if vlm:
        torch.testing.assert_close(model._prefix_tactile_keys(cache["prefix_pad_masks"]), roles == 2)
    saved = [layer.keys.clone() for layer in cache["past_key_values"].layers]
    calls = []
    original = model.vlm_with_expert.forward

    def capture(*args, **kwargs):
        calls.append(
            {
                key: kwargs[key].detach().clone() if kwargs[key] is not None else None
                for key in ("attention_mask", "position_ids", "noise_condition", "expert_condition")
            }
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(model.vlm_with_expert, "forward", capture)
    actions, noise, time = (
        torch.randn(2, 4, 6),
        torch.randn(2, 4, 6),
        torch.tensor([[0.1, 0.1, 0.8, 0.8]]).expand(2, -1),
    )
    result = model.denoise_step(
        cache["prefix_pad_masks"],
        cache["past_key_values"],
        time[..., None] * noise + (1 - time[..., None]) * actions,
        time,
        tactile_batch=batch,
        return_tactile_copy=ca0,
        apply_head_guidance=False,
    )
    raw, copy = result if ca0 else (result, None)
    loss = model([], [], lang, lang_mask, state, actions, noise, time, tactile_batch=batch)
    expected = (raw - (noise - actions)).square()
    if ca0:
        expected[:, 0] = 0.5 * (expected[:, 0] + (copy - (noise - actions)[:, 0]).square())
    torch.testing.assert_close(loss, expected)
    loss.mean().backward()
    assert model.tactile_encoder[0].weight.grad.abs().sum() > 0
    suffix_len = model._delay_token_count(4) + (1 if tcn else 3) + int(ca0) + 4
    assert calls[0]["position_ids"].shape[1] == suffix_len
    assert calls[0]["noise_condition"].shape[1] == suffix_len
    if delay_route == "sincos":
        assert calls[0]["expert_condition"].shape[1] == suffix_len
    torch.testing.assert_close(calls[0]["attention_mask"], calls[1]["attention_mask"][:, -suffix_len:])
    torch.testing.assert_close(calls[0]["position_ids"], calls[1]["position_ids"][:, -suffix_len:])
    assert (calls[0]["position_ids"][:, -4:].diff() == 1).all()
    if not ca0 and vlm:
        mask = calls[0]["attention_mask"]
        keys = roles == 2
        assert not mask[:, -4:-2, :12].masked_select(keys[:, None].expand(-1, 2, -1)).any()
        assert mask[:, -2:, :12].masked_select(keys[:, None].expand(-1, 2, -1)).all()
    for old, layer in zip(saved, cache["past_key_values"].layers, strict=True):
        torch.testing.assert_close(old, layer.keys)
    if not ca0:
        with pytest.raises(ValueError, match="use_tactile_ca0=True"):
            model.denoise_step(
                cache["prefix_pad_masks"],
                cache["past_key_values"],
                actions,
                time,
                tactile_batch=batch,
                return_tactile_copy=True,
            )


def test_ca0_config_default_validation_and_round_trip(tmp_path):
    assert SmolVLAConfig().use_tactile_ca0
    assert not SmolVLAConfig().fm_tactile_all_actions
    config = SmolVLAConfig(
        use_tactile=True, use_fm_tactile_token=True, use_tactile_ca0=True, fm_tactile_all_actions=True
    )
    config.save_pretrained(tmp_path)
    loaded = SmolVLAConfig.from_pretrained(tmp_path)
    assert loaded.use_tactile_ca0
    assert loaded.fm_tactile_all_actions
    # Checkpoints that predate the toggles use the default CA0-only routing.
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    del saved["use_tactile_ca0"]
    del saved["fm_tactile_all_actions"]
    path.write_text(json.dumps(saved))
    loaded = SmolVLAConfig.from_pretrained(tmp_path)
    assert loaded.use_tactile_ca0
    assert not loaded.fm_tactile_all_actions
    replace(config, use_tactile_ca0=False).save_pretrained(tmp_path)
    assert not SmolVLAConfig.from_pretrained(tmp_path).use_tactile_ca0
    with pytest.raises(ValueError, match="use_tactile_cfg_gate requires use_tactile_ca0"):
        replace(config, use_tactile_ca0=False, use_tactile_cfg_gate=True)


def test_no_ca0_bf16_training_and_return_guards(make_model):
    template = make_model(tactile=True, tactile_cross_attention=True, enabled=False)
    model = VLAFlowMatching(replace(template.config, use_tactile_ca0=False))
    lang = torch.tensor([[1, 2]])
    batch = {f"observation.tactile.{side}_marker": torch.randn(1, 2, 3, 2) for side in ("left", "right")}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(
            [],
            [],
            lang,
            torch.ones_like(lang, dtype=torch.bool),
            torch.randn(1, 6),
            torch.randn(1, 4, 6),
            tactile_batch=batch,
        ).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert model.tactile_encoder[0].weight.grad.abs().sum() > 0
    for mode in ("stream-head-tactile", "stream-head-tactile-cfg", "stream-head-mean-tactile-cfg"):
        with pytest.raises(ValueError, match="use_tactile_ca0=True"):
            model.stream_head_step_from_prefix({}, tactile_batch=batch, inference_mode=mode)

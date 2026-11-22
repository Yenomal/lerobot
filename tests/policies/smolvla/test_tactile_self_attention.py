"""Self/cross tactile receiver paths can be enabled independently."""

import json
from dataclasses import replace

import pytest
import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
from tests.policies.smolvla import test_delay_adarms

make_model = test_delay_adarms.make_model


@pytest.mark.parametrize("self_reads", [False, True])
@pytest.mark.parametrize("cross_reads", [False, True])
@pytest.mark.parametrize("ca0", [False, True])
@pytest.mark.parametrize("fingers", [False, True])
@pytest.mark.parametrize("amp", [False, True])
def test_self_cross_paths_masks_outputs_gradients_and_cached_parity(
    make_model, self_reads, cross_reads, ca0, fingers, amp
):
    torch.manual_seed(67)
    template = make_model(
        tactile=True, enabled=False, tactile_cross_attention=cross_reads, modality_count_bias=True
    )
    model = VLAFlowMatching(
        replace(
            template.config,
            use_tactile_self_attention=self_reads,
            use_tactile_ca0=ca0,
            use_tactile_finger_tokens=fingers,
            use_tactile_causal_attention=True,
            fm_tactile_token_history_length=3,
        )
    )
    for layer in model.vlm_with_expert.lm_expert.layers:
        torch.nn.init.normal_(layer.self_attn.o_proj.weight, std=0.5)
    lang, valid, state = torch.tensor([[1, 2], [3, 4]]), torch.ones(2, 2, dtype=torch.bool), torch.randn(2, 6)
    data = {f"observation.tactile.{side}_marker": torch.randn(2, 3, 2, 3, 2) for side in ("left", "right")}
    actions, noise, time = torch.randn(2, 4, 6), torch.randn(2, 4, 6), torch.full((2, 4), 0.5)
    captures = []
    expert = model.vlm_with_expert
    original = expert.get_attention_interface()

    def attention(mask, batch_size, head_dim, query, key, value, bias=None):
        captures.append((mask.clone(), key.shape[1]))
        return original(mask, batch_size, head_dim, query, key, value, bias)

    expert.get_attention_interface = lambda: attention
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        cache = model.encode_prefix([], [], lang, valid, state)
        prefix_len = cache["prefix_pad_masks"].shape[1]
        saved = [layer.keys.clone() for layer in cache["past_key_values"].layers]
        captures.clear()

        def predict(batch):
            result = model.denoise_step(
                cache["prefix_pad_masks"],
                cache["past_key_values"],
                0.5 * (actions + noise),
                time,
                tactile_batch=batch,
                return_tactile_copy=ca0,
                apply_head_guidance=False,
            )
            return result if ca0 else (result, None)

        raw, copy = predict(data)
        tactile_count = 6 if fingers else 3
        receiver = tactile_count  # CA0 or A0.
        self_mask = captures[0][0]
        assert self_mask[:, receiver, prefix_len : prefix_len + tactile_count].all().item() == self_reads
        assert not self_mask[:, receiver + (1 if ca0 else 2) :, prefix_len : prefix_len + tactile_count].any()
        times = torch.arange(tactile_count) // (2 if fingers else 1)
        expected_internal = times[:, None] >= times[None]
        torch.testing.assert_close(
            self_mask[:, :tactile_count, prefix_len : prefix_len + tactile_count],
            expected_internal[None].expand(2, -1, -1),
        )
        cross_mask, cross_keys = captures[1]
        assert cross_keys == prefix_len + (tactile_count if cross_reads else 0)
        if cross_reads:
            assert cross_mask[:, receiver, prefix_len:].all()
        if ca0:
            assert not self_mask[:, receiver + 1 :, prefix_len : prefix_len + tactile_count].any()
        changed_raw, changed_copy = predict({key: value * -10 for key, value in data.items()})
        first, changed_first = (copy, changed_copy) if ca0 else (raw[:, 0], changed_raw[:, 0])
        if self_reads or cross_reads:
            assert not torch.allclose(first, changed_first)
        else:
            torch.testing.assert_close(first, changed_first, rtol=0, atol=0)
        loss = model([], [], lang, valid, state, actions, noise, time, tactile_batch=data)
    expected_loss = (raw - (noise - actions)).square()
    if ca0:
        expected_loss[:, 0] = 0.5 * (expected_loss[:, 0] + (copy - (noise - actions)[:, 0]).square())
    torch.testing.assert_close(loss, expected_loss)
    for before, layer in zip(saved, cache["past_key_values"].layers, strict=True):
        torch.testing.assert_close(before, layer.keys, rtol=0, atol=0)
    loss.mean().backward()
    gradient = model.tactile_encoder[0].weight.grad
    if self_reads or cross_reads:
        assert gradient.abs().sum() > 0
    else:
        assert gradient is None or not gradient.any()


def test_self_attention_legacy_default_and_config_round_trip(tmp_path):
    assert SmolVLAConfig().use_tactile_self_attention
    config = SmolVLAConfig(use_tactile=True, use_fm_tactile_token=True, use_tactile_self_attention=False)
    config.save_pretrained(tmp_path)
    assert not SmolVLAConfig.from_pretrained(tmp_path).use_tactile_self_attention
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    del saved["use_tactile_self_attention"]
    path.write_text(json.dumps(saved))
    assert SmolVLAConfig.from_pretrained(tmp_path).use_tactile_self_attention

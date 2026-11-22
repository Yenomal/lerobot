"""Independent temporal encoding per finger and shared RoPE time positions."""

from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.policies.smolvla.smolvlm_with_expert import FlowTimeRMSNorm
from tests.policies.smolvla import test_delay_adarms

make_model = test_delay_adarms.make_model


def features():
    return {
        f"observation.tactile.{side}_marker": PolicyFeature(type=FeatureType.STATE, shape=(2, 3, 2))
        for side in ("left", "right")
    }


def batch():
    return {key: torch.randn(2, 3, 2, 3, 2) for key in features()}


@pytest.mark.parametrize("delta", [False, True])
@pytest.mark.parametrize("tcn", [False, True])
def test_finger_identity_shared_encoder_independent_temporal_convolution_and_checkpoint(
    make_model, delta, tcn, tmp_path
):
    torch.manual_seed(65)
    model = VLAFlowMatching(
        replace(
            make_model(tactile=True, enabled=False).config,
            use_tactile_finger_tokens=True,
            use_tactile_delta=delta,
            use_vlm_tactile_token=True,
            use_tactile_token_history=tcn,
            tactile_token_history_length=3,
            input_features=features(),
        )
    )
    data = batch()
    data["observation.tactile.right_marker"] = data["observation.tactile.left_marker"].clone()
    for vlm, prefix in ((False, ""), (True, "vlm_")):
        tokens = model._encode_tactile(data, vlm=vlm)
        width = model.vlm_with_expert.expert_hidden_size
        paired = tokens.reshape(2, -1, 2, width)
        identity = getattr(model, f"{prefix}tactile_finger_embedding").weight
        # Same markers must use the same MLP/TCN; only ID embedding differs.
        torch.testing.assert_close(paired[:, :, 0] - identity[0], paired[:, :, 1] - identity[1])
        assert paired.shape[1] == (1 if tcn else 3)
        assert getattr(model, f"{prefix}tactile_encoder")[0].in_features == (6 if delta else 12)
        changed = {k: v.clone() for k, v in data.items()}
        changed["observation.tactile.right_marker"][:, :, 1] *= -10
        changed_tokens = model._encode_tactile(changed, vlm=vlm).reshape_as(paired)
        torch.testing.assert_close(paired[:, :, 0], changed_tokens[:, :, 0], rtol=0, atol=0)
        assert not torch.allclose(paired[:, :, 1], changed_tokens[:, :, 1])
    assert model.tactile_finger_embedding.weight is not model.vlm_tactile_finger_embedding.weight
    policy = object.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.model = VLAFlowMatching(model.config)
    policy.config = model.config
    assert policy.initialize_lazy_parameters()
    assert policy.model.tactile_encoder[0].in_features == (6 if delta else 12)
    assert policy.model.vlm_tactile_encoder[0].in_features == (6 if delta else 12)
    policy.model = model
    saved = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    path = str(tmp_path / "model.safetensors")
    save_file(saved, path)
    SmolVLAPolicy._load_as_safetensor(policy, path, "cpu", True)
    torch.testing.assert_close(
        policy.model.tactile_finger_embedding.weight, saved["model.tactile_finger_embedding.weight"]
    )
    assert {"model.tactile_finger_embedding", "model.vlm_tactile_finger_embedding"} <= set(
        policy._get_default_peft_targets()["modules_to_save"]
    )


@pytest.mark.parametrize("ca0", [False, True])
@pytest.mark.parametrize("tcn,causal", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("attention_mode", ["self_attn", "cross_attn"])
@pytest.mark.parametrize("amp", [False, True])
def test_finger_rope_masks_conditions_cache_parity_and_gradients(
    make_model, monkeypatch, ca0, tcn, causal, attention_mode, amp
):
    torch.manual_seed(66)
    model = VLAFlowMatching(
        replace(
            make_model(
                tactile=True,
                enabled=False,
                attention_mode=attention_mode,
                tactile_cross_attention=attention_mode == "cross_attn",
                delay_token=True,
                sincos=True,
                modality_count_bias=True,
                use_noise_adarms=True,
            ).config,
            use_tactile_finger_tokens=True,
            use_tactile_delta=True,
            use_vlm_tactile_token=True,
            use_tactile_ca0=ca0,
            use_tactile_token_history=tcn,
            use_tactile_causal_attention=causal,
            fm_tactile_token_history_length=3,
            vlm_tactile_token_history_length=2,
            vlm_context_reads_tactile=False,
            fm_tactile_reads_vlm_tactile=False,
            prefix_length=12,
        )
    )
    for layer in model.vlm_with_expert.lm_expert.layers:
        nn.init.normal_(layer.self_attn.o_proj.weight, std=0.5)
    for module in model.modules():
        if isinstance(module, FlowTimeRMSNorm):
            nn.init.normal_(module.projection.weight, std=0.2)
    captures = []
    original = model.vlm_with_expert.forward

    def capture(*args, **kwargs):
        captures.append(
            {
                k: kwargs.get(k)
                for k in ("attention_mask", "position_ids", "expert_condition", "noise_condition")
            }
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(model.vlm_with_expert, "forward", capture)
    data = batch()
    lang, valid, state = (
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([[True, False], [True, True]]),
        torch.randn(2, 6),
    )
    actions, noise, time = torch.randn(2, 4, 6), torch.randn(2, 4, 6), torch.full((2, 4), 0.5)
    delay = {"observation_delay_tokens": torch.tensor([0, 2])}
    fc, vc = (2, 2) if tcn else (6, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        cache = model.encode_prefix([], [], lang, valid, state, tactile_batch=data, **delay)
        prefix_positions = captures[-1]["position_ids"]
        torch.testing.assert_close(prefix_positions[:, 3 : 3 + vc : 2], prefix_positions[:, 4 : 3 + vc : 2])
        roles = model.embed_prefix([], [], lang, valid, state, tactile_batch=data)[2]
        torch.testing.assert_close(model._prefix_tactile_keys(cache["prefix_pad_masks"]), roles == 2)
        result = model.denoise_step(
            cache["prefix_pad_masks"],
            cache["past_key_values"],
            0.5 * (actions + noise),
            time,
            tactile_batch=data,
            return_tactile_copy=ca0,
            apply_head_guidance=False,
            **delay,
        )
        cached = captures[-1]
        loss = model([], [], lang, valid, state, actions, noise, time, tactile_batch=data, **delay)
    raw, copy = result if ca0 else (result, None)
    expected = (raw - (noise - actions)).square()
    if ca0:
        expected[:, 0] = 0.5 * (expected[:, 0] + (copy - (noise - actions)[:, 0]).square())
    torch.testing.assert_close(loss, expected)
    full = captures[-1]
    suffix_len = 4 + fc + int(ca0) + 4
    torch.testing.assert_close(cached["position_ids"], full["position_ids"][:, -suffix_len:])
    torch.testing.assert_close(cached["attention_mask"], full["attention_mask"][:, -suffix_len:])
    torch.testing.assert_close(
        cached["position_ids"][:, 4 : 4 + fc : 2], cached["position_ids"][:, 5 : 4 + fc : 2]
    )
    assert (cached["position_ids"][:, -4:].diff() == 1).all()
    assert cached["expert_condition"] is None  # Delay tokens still align without modulation.
    assert cached["noise_condition"].shape[1] == suffix_len
    torch.testing.assert_close(
        cached["noise_condition"][:, 4 : 4 + fc], cached["noise_condition"][:, :1].expand(-1, fc, -1)
    )
    expected_block = torch.ones(fc, fc, dtype=torch.bool)
    if causal:
        times = torch.arange(fc) // 2
        expected_block = times[:, None] >= times[None]
    torch.testing.assert_close(
        cached["attention_mask"][:, 4 : 4 + fc, 16 : 16 + fc], expected_block[None].expand(2, -1, -1)
    )
    loss.mean().backward()
    assert model.tactile_finger_embedding.weight.grad.abs().sum(-1).gt(0).all()
    assert model.vlm_tactile_finger_embedding.weight.grad.abs().sum(-1).gt(0).all()


def test_split_reconstruction_and_gate_pool_both_current_fingers(make_model):
    template = make_model(tactile=True, enabled=False)
    config = replace(
        template.config,
        use_tactile_finger_tokens=True,
        use_tactile_delta=True,
        use_tactile_reconstruction=True,
        input_features=features(),
    )
    model = VLAFlowMatching(config)
    data = batch()
    tokens = model._encode_tactile(data)
    latest = tokens[:, -2:]
    target = torch.stack([v[:, -1, 1].flatten(1) - v[:, -1, 0].flatten(1) for v in data.values()], 1)
    assert model.tactile_reconstruction[-1].out_features == 6
    lang = torch.ones(2, 2, dtype=torch.long)
    loss = model([], [], lang, lang.bool(), torch.randn(2, 6), torch.randn(2, 4, 6), tactile_batch=data)
    torch.testing.assert_close(
        loss[:, 0], (model.tactile_reconstruction(latest) - target).square().flatten(1)
    )
    gate = VLAFlowMatching(replace(config, use_tactile_reconstruction=False, use_tactile_cfg_gate=True))
    suffix = torch.randn(2, 11, 8)
    captured = []
    handle = gate.tactile_cfg_gate.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    gate._tactile_cfg_strength(suffix, 4)
    handle.remove()
    torch.testing.assert_close(captured[0], suffix[:, -7:-5].mean(1))


def test_finger_config_defaults_and_round_trip(tmp_path):
    assert not SmolVLAConfig().use_tactile_finger_tokens
    config = SmolVLAConfig(use_tactile=True, use_tactile_finger_tokens=True)
    config.save_pretrained(tmp_path)
    assert SmolVLAConfig.from_pretrained(tmp_path).use_tactile_finger_tokens
    with pytest.raises(ValueError, match="requires use_tactile=True"):
        SmolVLAConfig(use_tactile_finger_tokens=True)

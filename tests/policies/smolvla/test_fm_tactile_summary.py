# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import replace

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from tests.policies.smolvla.test_delay_adarms import make_model  # noqa: F401

LEFT = "observation.tactile.left_marker"
RIGHT = "observation.tactile.right_marker"


def test_summary_aux_switch_controls_future_labels_independently_of_weight():
    config = SmolVLAConfig(
        use_tactile=True,
        use_fm_tactile_token=True,
        use_fm_tactile_summary_token=True,
        fm_tactile_summary_aux_weight=1.0,
        use_fm_tactile_summary_aux_loss=False,
    )
    assert config.tactile_observation_delta_indices == [0]

    config.use_fm_tactile_summary_aux_loss = True
    assert config.tactile_observation_delta_indices == [0, 5, 10, 15]


@pytest.mark.parametrize("ca0", [False, True])
@pytest.mark.parametrize("summary_only", [False, True])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("all_actions", [False, True])
def test_summary_mask(ca0, summary_only, cached, all_actions):
    prefix, delay, history, actions = 4, 1, 4, 4
    start = prefix + delay
    summary = start + history
    first_action = summary + 1 + int(ca0)
    length = first_action + actions
    offset = prefix if cached else 0
    mask = VLAFlowMatching._apply_tactile_token_mask(
        torch.ones(1, length - offset, length, dtype=torch.bool),
        prefix_len=prefix,
        action_count=actions,
        query_offset=offset,
        delay_token=delay,
        tactile_token_count=history,
        tactile_frame_size=2,
        tactile_action_count=2,
        causal=True,
        use_ca0=ca0,
        has_summary=True,
        summary_only=summary_only,
        all_actions=all_actions,
    )[0]
    assert mask[summary - offset, :prefix].all()
    assert mask[summary - offset, start : summary + 1].all()
    assert not mask[summary - offset, prefix:start].any()
    assert not mask[summary - offset, summary + 1 :].any()
    assert not mask[start - offset : summary - offset, summary].any()
    assert mask[start - offset, start : start + 2].all()
    assert not mask[start - offset, start + 2 : summary].any()
    end = length if all_actions else first_action + (0 if ca0 else 2)
    receiver = slice(summary + 1 - offset, end - offset)
    assert mask[receiver, summary].all()
    assert (mask[receiver, start:summary] == (not summary_only)).all()
    assert not mask[receiver.stop :, start : summary + 1].any()


@pytest.mark.parametrize("ca0,tcn,split", [(False, False, False), (True, True, True)])
def test_summary_prediction_and_cached_inference(make_model, ca0, tcn, split):  # noqa: F811
    torch.manual_seed(8)
    config = replace(
        make_model(tactile=True, tactile_cross_attention=True, delay_token=True, sincos=True).config,
        use_tactile_ca0=ca0,
        use_tactile_delta=True,
        use_tactile_tcn=tcn,
        use_tactile_finger_tokens=split,
        fm_tactile_token_history_length=5,
        streaming_constant_weight=0.0,
        use_fm_tactile_summary_token=True,
        use_fm_tactile_summary_aux_loss=True,
        fm_tactile_summary_aux_weight=0.3,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            LEFT: PolicyFeature(type=FeatureType.STATE, shape=(2, 3, 2)),
            RIGHT: PolicyFeature(type=FeatureType.STATE, shape=(2, 3, 2)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    )
    model = VLAFlowMatching(config)
    assert config.tactile_observation_delta_indices == [-4, -3, -2, -1, 0, 5, 10, 15]
    state, actions = torch.randn(2, 6), torch.randn(2, 4, 6)
    lang = torch.tensor([[1, 2], [3, 4]])
    lang_mask = torch.ones_like(lang, dtype=torch.bool)
    noise, time = torch.randn_like(actions), torch.tensor([0.3, 0.6])
    observations = ([], [], lang, lang_mask, state)
    delay = {"observation_delay_tokens": torch.tensor([1, 2])}
    batch = {key: torch.randn(2, 8, 2, 3, 2) for key in (LEFT, RIGHT)}
    batch[f"{RIGHT}_is_pad"] = torch.tensor([[False] * 8, [False] * 6 + [True] * 2])
    history, targets, valid = model._split_tactile_prediction_batch(batch)
    torch.testing.assert_close(history[LEFT], batch[LEFT][:, :5])
    torch.testing.assert_close(history[RIGHT], batch[RIGHT][:, :5])
    marker = torch.cat(
        [(batch[key][:, 4:, 1] - batch[key][:, 4:, 0]).flatten(2) for key in (LEFT, RIGHT)], -1
    )
    torch.testing.assert_close(targets[:, 0], marker[:, 0])
    torch.testing.assert_close(targets[:, 1:], marker[:, 1:] - marker[:, :1])
    assert valid.tolist() == [[True] * 4, [True, True, False, False]]

    loss, aux, _ = model(*observations, actions, noise, time, tactile_batch=batch, **delay)
    changed_targets = {**batch, LEFT: batch[LEFT].clone()}
    changed_targets[LEFT][:, 5:, 1] += 25
    changed_loss, changed_aux, _ = model(
        *observations, actions, noise, time, tactile_batch=changed_targets, **delay
    )
    torch.testing.assert_close(loss, changed_loss)
    torch.testing.assert_close(aux[:, 0], changed_aux[:, 0])
    assert not torch.allclose(aux[0, 1:], changed_aux[0, 1:])
    assert (aux[1, 2:] == 0).all()

    cache = model.encode_prefix(*observations)
    x_t = torch.lerp(actions, noise, time[:, None, None])
    result = model.denoise_step(
        cache["prefix_pad_masks"],
        cache["past_key_values"],
        x_t,
        time,
        tactile_batch=history,
        apply_head_guidance=False,
        return_tactile_copy=ca0,
        **delay,
    )
    velocity, copy = result if ca0 else (result, None)
    expected = (noise - actions - velocity).square()
    if copy is not None:
        expected[:, 0] = (expected[:, 0] + (noise[:, 0] - actions[:, 0] - copy).square()) / 2
    torch.testing.assert_close(loss, expected)

    policy = object.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config, policy.model = config, model
    policy.prepare_images = lambda _: ([], [])
    policy_batch = {
        **batch,
        **delay,
        OBS_STATE: state,
        OBS_LANGUAGE_TOKENS: lang,
        OBS_LANGUAGE_ATTENTION_MASK: lang_mask,
        ACTION: actions,
    }
    per_sample, metrics = policy(policy_batch, noise, time, reduction="none")
    expected_aux = aux.sum(1) / valid.sum(1)
    torch.testing.assert_close(per_sample, loss.mean((1, 2)) + 0.3 * expected_aux)
    torch.testing.assert_close(metrics["action_loss"], loss.mean().item(), rtol=0, atol=1e-6)
    assert metrics["tactile/summary_aux_loss"] == pytest.approx((0.3 * expected_aux).mean().item())
    mean, _ = policy(policy_batch, noise, time)
    torch.testing.assert_close(mean, per_sample.mean())
    assert metrics["tactile/summary_prediction_mse_15"] == pytest.approx(aux[0, 3].item())


@pytest.mark.parametrize("attention_mode", ["self_attn", "cross_attn"])
def test_summary_detaches_vlm_only_for_tactile_queries(make_model, attention_mode):  # noqa: F811
    torch.manual_seed(5)
    cross = attention_mode == "cross_attn"
    config = replace(
        make_model(attention_mode, tactile=True, tactile_cross_attention=cross).config,
        self_attn_every_n_layers=0,
        use_tactile_ca0=False,
        use_fm_tactile_summary_token=True,
        use_fm_tactile_summary_aux_loss=True,
        fm_tactile_summary_aux_weight=1.0,
        fm_tactile_stop_gradient_to_vlm=True,
        input_features={key: PolicyFeature(type=FeatureType.STATE, shape=(2, 3, 2)) for key in (LEFT, RIGHT)},
    )
    model = VLAFlowMatching(config)
    state, actions = torch.randn(2, 6), torch.randn(2, 4, 6)
    lang = torch.tensor([[1, 2], [3, 4]])
    noise, time = torch.randn_like(actions), torch.tensor([0.2, 0.8])
    batch = {key: torch.randn(2, 4, 2, 3, 2) for key in (LEFT, RIGHT)}

    def run():
        return model(
            [],
            [],
            lang,
            torch.ones_like(lang, dtype=torch.bool),
            state,
            actions,
            noise,
            time,
            tactile_batch=batch,
        )

    prefix_weight = model.state_proj.weight
    embedded_language = model.vlm_with_expert.get_vlm_model().text_model.embed_tokens.weight
    embedded_language.requires_grad_(True)
    action, auxiliary, _ = run()
    tactile_weight = model.tactile_encoder[0].weight
    token_weight = model.tactile_summary_token.weight
    decoder_weight = model.tactile_summary_decoder[-1].weight
    parameters = (prefix_weight, embedded_language, tactile_weight, token_weight, decoder_weight)
    gradients = torch.autograd.grad(auxiliary.mean(), parameters, retain_graph=True, allow_unused=True)
    assert all(grad is None or not grad.any() for grad in gradients[:2])
    assert all(grad is not None and grad.abs().sum() > 0 for grad in gradients[2:])
    if cross:
        projection = model.vlm_with_expert.lm_expert.layers[0].self_attn.v_proj.weight
        assert torch.autograd.grad(auxiliary.mean(), projection, retain_graph=True)[0].abs().sum() > 0
    action_gradients = torch.autograd.grad(action.mean(), (prefix_weight, tactile_weight, token_weight))
    assert all(grad.abs().sum() > 0 for grad in action_gradients)
    model.vlm_with_expert.fm_tactile_stop_gradient_to_vlm = False
    full_action, full_aux, _ = run()
    torch.testing.assert_close(action, full_action)
    torch.testing.assert_close(auxiliary, full_aux)
    assert torch.autograd.grad(full_aux.mean(), prefix_weight)[0].abs().sum() > 0

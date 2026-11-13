from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("tactile", [False, True])
def test_delay_visibility_preserves_tactile_isolation(cached, tactile):
    prefix_len, horizon = 4, 5
    action_start = prefix_len + 1 + (2 if tactile else 0)
    length = action_start + horizon
    valid = torch.ones(1, length, dtype=torch.bool)
    valid[:, 1] = False  # A padded VLM key must remain invisible.
    boundaries = torch.zeros_like(valid)
    boundaries[:, prefix_len] = True
    boundaries[:, action_start] = True
    mask = make_att_2d_masks(valid, boundaries)
    offset = prefix_len if cached else 0
    mask = mask[:, offset:]
    mask = VLAFlowMatching._apply_delay_token_mask(
        mask,
        prefix_len=prefix_len,
        action_count=horizon,
        action_start_offset=action_start - prefix_len,
        query_offset=offset,
    )
    if tactile:
        mask = VLAFlowMatching._apply_tactile_token_mask(
            mask,
            prefix_len=prefix_len,
            action_count=horizon,
            query_offset=offset,
            delay_token=True,
        )
    delay_row = mask[0, prefix_len - offset]
    assert delay_row[:prefix_len].tolist() == [True, False, True, True]
    assert delay_row[prefix_len]
    assert not delay_row[prefix_len + 1 :].any()
    assert mask[0, action_start - offset :, prefix_len].all()
    assert not mask[:, :, 1].any()
    if not cached:
        assert not mask[:, :prefix_len, prefix_len:].any()
    if tactile:
        tactile_index, copy_index = prefix_len + 1, prefix_len + 2
        assert not mask[0, action_start - offset :, tactile_index : copy_index + 1].any()
        assert mask[0, copy_index - offset, tactile_index]
        assert mask[0, copy_index - offset, prefix_len]


def test_split_policy_forwards_current_delay_instead_of_cached_delay():
    seen = {}

    def sample(cache, **kwargs):
        seen.update(kwargs)
        return torch.zeros(1, 5, 2)

    policy = SimpleNamespace(
        eval=lambda: None,
        _prepare_batch=lambda batch: batch,
        model=SimpleNamespace(sample_actions_from_prefix=sample),
        _postprocess_action_chunk=lambda actions: actions,
    )
    current_delay = torch.tensor([3])
    SmolVLAPolicy.predict_action_chunk_from_prefix(
        policy,
        {"observation_delay_tokens": current_delay},
        {"observation_delay_tokens": torch.tensor([0])},
    )
    assert seen["observation_delay_tokens"] is current_delay


def test_delay_changes_only_condition_token_before_attention():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        streaming=True,
        streaming_attention_mode="bidirectional",
        chunk_size=5,
        observation_delay_max_tokens=4,
        min_period=0.004,
        max_period=4.0,
        use_tactile_token=False,
        use_noise_embedding=True,
    )
    model.vlm_with_expert = SimpleNamespace(expert_hidden_size=4)
    model._delay_max = 4
    model.action_in_proj = torch.nn.Linear(2, 4)
    model.action_time_mlp_in = torch.nn.Linear(8, 4)
    model.action_time_mlp_out = torch.nn.Linear(4, 4)
    model.observation_delay_embedding = None
    model.observation_delay_adarms = None
    model.observation_delay_token_embedding = torch.nn.Embedding(5, 4)
    noise = torch.randn(1, 5, 2)
    time = torch.linspace(0.1, 0.9, 5)[None]
    first, _, _ = model.embed_suffix(noise, time, observation_delay_tokens=torch.tensor([0]))
    second, _, _ = model.embed_suffix(noise, time, observation_delay_tokens=torch.tensor([3]))
    assert first.shape == (1, 6, 4)
    torch.testing.assert_close(first[:, 1:], second[:, 1:])
    assert not torch.equal(first[:, 0], second[:, 0])
    second[:, 0].sum().backward()
    assert model.observation_delay_token_embedding.weight.grad[3].abs().sum() > 0


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("tactile", [False, True])
def test_per_action_delay_tokens_read_only_prefix_self_and_match_actions(cached, tactile):
    prefix_len, horizon = 3, 4
    action_start = prefix_len + horizon + (2 if tactile else 0)
    length = action_start + horizon
    valid = torch.ones(1, length, dtype=torch.bool)
    valid[:, 1] = False
    boundaries = torch.zeros_like(valid)
    boundaries[:, prefix_len] = True
    boundaries[:, action_start] = True
    offset = prefix_len if cached else 0
    mask = make_att_2d_masks(valid, boundaries)[:, offset:]
    mask = VLAFlowMatching._apply_delay_token_mask(
        mask,
        prefix_len=prefix_len,
        action_count=horizon,
        action_start_offset=action_start - prefix_len,
        query_offset=offset,
        delay_token_count=horizon,
    )
    if tactile:
        mask = VLAFlowMatching._apply_tactile_token_mask(
            mask,
            prefix_len=prefix_len,
            action_count=horizon,
            query_offset=offset,
            delay_token=horizon,
        )
    for i in range(horizon):
        row = mask[0, prefix_len + i - offset]
        assert row[:prefix_len].tolist() == [True, False, True]
        torch.testing.assert_close(
            row[prefix_len : prefix_len + horizon], torch.eye(horizon, dtype=torch.bool)[i]
        )
        assert not row[prefix_len + horizon :].any()
        action_row = mask[0, action_start + i - offset]
        torch.testing.assert_close(
            action_row[prefix_len : prefix_len + horizon], torch.eye(horizon, dtype=torch.bool)[i]
        )
    assert not mask[:, :, 1].any()
    if not cached:
        assert not mask[:, :prefix_len, prefix_len:].any()
    if tactile:
        tactile_index = prefix_len + horizon
        assert not mask[0, tactile_index - offset, prefix_len : prefix_len + horizon].any()
        copy_row = mask[0, tactile_index + 1 - offset]
        assert copy_row[prefix_len]
        assert not copy_row[prefix_len + 1 : prefix_len + horizon].any()
        assert not mask[0, action_start - offset :, tactile_index : tactile_index + 2].any()

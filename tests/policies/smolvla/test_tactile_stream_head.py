"""Tactile head guidance keeps CA0 separate from the raw action history."""

from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

TACTILE_MODES = (
    "stream-head-tactile",
    "stream-head-tactile-cfg",
    "stream-head-mean-tactile-cfg",
)


@pytest.mark.parametrize("mode", TACTILE_MODES)
@pytest.mark.parametrize("strength", [0.0, 1.0, 1.5])
def test_tactile_guidance_changes_only_position_zero_and_preserves_raw_history(mode, strength):
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        chunk_size=6, streaming=True, streaming_chunk_size=3, action_dim=2, use_tactile_token=True
    )
    model.tactile_adarms = None
    model._encode_tactile = lambda batch: torch.ones(2, 2)
    model.sample_noise = lambda shape, device: torch.zeros(shape, device=device)
    raw = torch.arange(2.0, 14.0, 2.0)[None, :, None].expand(2, 6, 2).clone()
    copy = torch.tensor([[10.0, 14.0], [20.0, 24.0]])
    model.denoise_step = lambda **kwargs: (raw, copy)
    # The next step shifts the old position 1 into position 0, carrying its history.
    history = torch.tensor([8.0, 10.0])[None, None, :, None].expand(2, 6, 2, 2).clone()
    model._stream_head_state = {
        "action_window": torch.zeros(2, 6, 2),
        "timestep_window": torch.tensor([[0.5, 0.5, 0.5, 1.0, 1.0, 1.0]]).expand(2, -1),
        "history": history,
        "history_valid": torch.ones(2, 6, 2, dtype=torch.bool),
    }
    prefix = {"prefix_pad_masks": torch.ones(2, 1, dtype=torch.bool), "past_key_values": None}
    actions = model.stream_head_step_from_prefix(
        prefix, tactile_batch={}, inference_mode=mode, tactile_cfg_lambda=strength
    )
    if mode == "stream-head-tactile":
        expected = copy
    else:
        base = (18.0 + raw[:, 0]) / 3.0 if mode == "stream-head-mean-tactile-cfg" else raw[:, 0]
        expected = base + strength * (copy - raw[:, 0])
    torch.testing.assert_close(actions[:, 0], -0.5 * expected)
    other_head = (18.0 + raw[:, 3]) / 3.0 if mode == "stream-head-mean-tactile-cfg" else raw[:, 3]
    torch.testing.assert_close(actions[:, 3], -0.5 * other_head)
    torch.testing.assert_close(actions[:, [1, 2, 4, 5]], torch.zeros(2, 4, 2))
    torch.testing.assert_close(model._stream_head_state["history"][:, :, -1], raw)
    torch.testing.assert_close(raw[:, 0], torch.full((2, 2), 2.0))


@pytest.mark.parametrize("mode", TACTILE_MODES)
@pytest.mark.parametrize("entrypoint", ["sample_actions_from_prefix", "stream_head_step_from_prefix"])
def test_tactile_modes_require_enabled_token_and_observation(mode, entrypoint):
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(use_tactile_token=False, chunk_size=6, action_dim=2)
    prefix = {"prefix_pad_masks": torch.ones(1, 1, dtype=torch.bool), "past_key_values": None}
    kwargs = {"noise": torch.zeros(1, 6, 2), "inference_mode": mode}
    with pytest.raises(ValueError, match="requires use_tactile_token=True"):
        getattr(model, entrypoint)(prefix, **kwargs)
    model.config.use_tactile_token = True
    with pytest.raises(ValueError, match="requires a tactile observation"):
        getattr(model, entrypoint)(prefix, **kwargs)


@pytest.mark.parametrize("delay", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_denoise_returns_ca0_separately_with_or_without_delay_token(delay, cached):
    from transformers import DynamicCache

    model = VLAFlowMatching.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(chunk_size=3, use_tactile_token=True, use_cache=cached)
    model.observation_delay_token_embedding = object() if delay else None
    model.observation_delay_adarms = None
    # Expert outputs use distinct values for T, CA0 and the real actions.
    suffix = torch.tensor([90.0, 100.0, 2.0, 4.0, 6.0]).reshape(1, 5, 1)
    if delay:
        suffix = torch.cat([torch.full((1, 1, 1), 80.0), suffix], dim=1)
    model.embed_suffix = lambda *args, **kwargs: (
        suffix,
        torch.ones(1, suffix.shape[1], dtype=torch.bool),
        torch.ones(1, suffix.shape[1], dtype=torch.bool),
    )
    keys = torch.arange(2.0).reshape(1, 1, 2, 1)
    cache = DynamicCache(ddp_cache_data=((keys, keys.clone()),)) if cached else None

    def forward(**kwargs):
        if cached:
            # Every denoise call must start from the same clean prefix even
            # when the worker retains the DynamicCache across requests.
            assert cache.get_seq_length() == 2
            torch.testing.assert_close(cache.layers[0].keys, keys)
            cache.update(suffix[:, None], suffix[:, None], 0)
        return (None, suffix), cache

    model.vlm_with_expert = SimpleNamespace(forward=forward)
    model.action_out_proj = torch.nn.Identity()
    kwargs = {
        "prefix_pad_masks": torch.ones(1, 2, dtype=torch.bool),
        "past_key_values": cache,
        "x_t": torch.zeros(1, 3, 1),
        "timestep": torch.ones(1, 3),
        "tactile_embedding": torch.ones(1, 1),
        "apply_head_guidance": False,
    }
    raw, copy = model.denoise_step(**kwargs, return_tactile_copy=True)
    torch.testing.assert_close(raw, torch.tensor([[[2.0], [4.0], [6.0]]]))
    torch.testing.assert_close(copy, torch.tensor([[100.0]]))
    torch.testing.assert_close(model.denoise_step(**kwargs), raw)
    if cached:
        assert cache.get_seq_length() == 2
        torch.testing.assert_close(cache.layers[0].keys, keys)


@pytest.mark.parametrize("mode, expected", list(zip(TACTILE_MODES, [-10.0, -15.0, -17.0], strict=True)))
def test_tactile_modes_initialize_and_carry_action_history_across_steps(mode, expected):
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        chunk_size=5,
        streaming=True,
        streaming_chunk_size=5,
        action_dim=2,
        num_steps=1,
        use_tactile_token=True,
    )
    model._stream_head_state = None
    model.tactile_adarms = None
    model._encode_tactile = lambda batch: torch.ones(1, 2)
    model.sample_noise = lambda shape, device: torch.zeros(shape, device=device)

    def denoise(*, x_t, return_tactile_copy=False, **kwargs):
        raw = torch.arange(5.0)[None, :, None].expand_as(x_t)
        return (raw, torch.full((1, 2), 10.0)) if return_tactile_copy else raw

    model.denoise_step = denoise
    prefix = {"prefix_pad_masks": torch.ones(1, 1, dtype=torch.bool), "past_key_values": None}
    for _ in range(5):
        actions = model.stream_head_step_from_prefix(prefix, tactile_batch={}, inference_mode=mode)
    # The token that began at position 4 now reaches 0 with raw fields 4,3,2,1,0.
    torch.testing.assert_close(actions[:, 0], torch.full((1, 2), expected))

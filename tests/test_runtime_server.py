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

"""Unit tests for server-owned action playback and its small wire helpers."""

import asyncio
from contextlib import asynccontextmanager

import numpy as np
import pytest

from lerobot.runtime_protocol import acknowledge, action_body, observation, read_message, reset, write_message
from lerobot.runtime_server import RuntimeServer, RuntimeSession
from lerobot.transport.protocol import MessageType, decode_message, encode_message
from tests.inference.policy_helpers import ConformingPolicy, runner_for, tiny_config


def test_runtime_session_replays_pending_action_until_ack_and_queues_the_horizon():
    policy = ConformingPolicy(tiny_config())
    calls = 0
    predict_original = policy.predict_action_chunk

    def predict(batch, **kwargs):
        nonlocal calls
        calls += 1
        return predict_original(batch, **kwargs)

    policy.predict_action_chunk = predict
    runtime = RuntimeSession(runner_for(policy), queue_size=8)
    features = {
        "observation.state": np.zeros(3, dtype=np.float32),
        "observation.environment_state": np.zeros(3, dtype=np.float32),
    }

    first = runtime.infer(features, "task", 1)
    replay = runtime.infer(features, "task", 2)
    assert replay["action_id"] == first["action_id"]
    assert replay["replayed"]
    assert calls == 1

    runtime.acknowledge(first["action_id"])
    second = runtime.infer(features, "task", 2)
    assert second["action"].tolist() != first["action"].tolist()
    assert calls == 1


def test_runtime_serves_two_sessions_and_scopes_ack_reset_and_disconnect():
    pytest.importorskip("msgpack")
    runtime = RuntimeServer(runner_for(ConformingPolicy(tiny_config())), "127.0.0.1", 0, 8)

    async def scenario():
        server = await asyncio.start_server(runtime._connection, runtime.host, runtime.port)
        port = server.sockets[0].getsockname()[1]
        features = {
            "observation.state": np.zeros(3, dtype=np.float32),
            "observation.environment_state": np.zeros(3, dtype=np.float32),
        }

        @asynccontextmanager
        async def connect():
            reader, writer = await asyncio.open_connection(runtime.host, port)

            async def send(message):
                await write_message(writer, message)
                reply = await read_message(reader)
                assert reply.session_id == message.session_id
                assert reply.request_id == message.request_id
                return reply

            try:
                yield send
            finally:
                writer.close()
                await writer.wait_closed()

        async with server, connect() as b:
            async with connect() as a:
                a_first, b_first = await asyncio.gather(
                    a(observation("robot-a", 0, features)), b(observation("robot-b", 0, features))
                )
                assert action_body(a_first)["action_id"] == action_body(b_first)["action_id"] == 0
                async with connect() as duplicate:
                    busy = await duplicate(reset("robot-b"))
                    assert busy.body["code"] == "busy"
                assert (await a(acknowledge("robot-a", 0))).message_type is MessageType.ACK
                assert action_body(await a(observation("robot-a", 1, features)))["action_id"] == 1
                assert (await a(reset("robot-a"))).message_type is MessageType.ACK
                assert action_body(await a(observation("robot-a", 0, features)))["action_id"] == 0
            replay = await b(observation("robot-b", 0, features))
            assert action_body(replay)["replayed"]
            np.testing.assert_array_equal(action_body(replay)["action"], action_body(b_first)["action"])
            assert (await b(acknowledge("robot-b", 0))).message_type is MessageType.ACK
            assert action_body(await b(observation("robot-b", 1, features)))["action_id"] == 1

    asyncio.run(asyncio.wait_for(scenario(), timeout=10))


def test_runtime_task_change_drops_old_plan_after_acknowledgement():
    policy = ConformingPolicy(tiny_config())
    calls = 0
    predict_original = policy.predict_action_chunk

    def predict(batch, **kwargs):
        nonlocal calls
        calls += 1
        return predict_original(batch, **kwargs)

    policy.predict_action_chunk = predict
    runtime = RuntimeSession(runner_for(policy), queue_size=8)
    features = {
        "observation.state": np.zeros(3, dtype=np.float32),
        "observation.environment_state": np.zeros(3, dtype=np.float32),
    }
    action = runtime.infer(features, "first", 1)
    runtime.acknowledge(action["action_id"])
    runtime.infer(features, "second", 2)
    assert calls == 2


def test_runtime_session_uses_split_prefix_and_refreshes_it_as_the_window_advances():
    policy = ConformingPolicy(tiny_config())
    policy.config.streaming = True
    policy.config.streaming_chunk_size = 2
    prefix_calls = 0
    fm_calls = 0

    def encode_prefix(batch):
        nonlocal prefix_calls
        prefix_calls += 1
        return {"prefix": prefix_calls}

    def predict_from_prefix(batch, prefix_cache, **kwargs):
        nonlocal fm_calls
        fm_calls += 1
        return policy.predict_action_chunk(batch, **kwargs)

    policy.encode_prefix = encode_prefix
    policy.predict_action_chunk_from_prefix = predict_from_prefix
    runtime = RuntimeSession(
        runner_for(policy),
        queue_size=1,
        streaming_horizon=4,
        streaming_watermark=2,
    )
    features = {
        "observation.state": np.zeros(3, dtype=np.float32),
        "observation.environment_state": np.zeros(3, dtype=np.float32),
    }

    first = runtime.infer(features, "task", 1)
    runtime.acknowledge(first["action_id"])
    second = runtime.infer(features, "task", 2)
    runtime.acknowledge(second["action_id"])
    third = runtime.infer(features, "task", 3)
    runtime.acknowledge(third["action_id"])
    runtime.close()

    assert prefix_calls >= 2
    assert fm_calls >= 2
    assert third["streaming"]["refresh_after"] == 2


def test_runtime_protocol_round_trip_keeps_observation_arrays_and_ack_identity():
    pytest.importorskip("msgpack")
    message = observation("robot-a", 4, {"state": np.array([1, 2], dtype=np.float32)}, task="move")
    decoded = decode_message(encode_message(message))
    assert decoded.message_type is MessageType.OBSERVATION
    assert decoded.body["features"]["state"].tolist() == [1.0, 2.0]
    ack = acknowledge("robot-a", 9)
    assert (
        action_body(
            decoded.reply(
                MessageType.ACTION,
                {"op": "action", "action_id": 9, "action": np.array([0.5], dtype=np.float32)},
            )
        )["action_id"]
        == 9
    )
    with pytest.raises(ValueError):
        action_body(ack)


@pytest.mark.parametrize(
    "mode", ["stream-head-tactile", "stream-head-tactile-cfg", "stream-head-mean-tactile-cfg"]
)
def test_runtime_tactile_modes_validate_config_and_forward_strength_each_step(mode):
    policy = ConformingPolicy(tiny_config())
    runner = runner_for(policy)
    with pytest.raises(ValueError, match="requires use_fm_tactile_token=True"):
        RuntimeSession(runner, queue_size=8, inference_mode=mode)
    policy.config.use_fm_tactile_token = True
    with pytest.raises(ValueError, match="requires split streaming inference"):
        RuntimeSession(runner, queue_size=8, inference_mode=mode)
    policy.config.streaming = True
    seen = []
    policy.encode_prefix = lambda batch: {}

    def predict_from_prefix(batch, cache, **kwargs):
        seen.append(kwargs)
        return policy.predict_action_chunk(batch)

    policy.predict_action_chunk_from_prefix = predict_from_prefix
    runtime = RuntimeSession(runner, queue_size=8, inference_mode=mode, tactile_cfg_lambda=2.25)
    features = {
        "observation.state": np.zeros(3, dtype=np.float32),
        "observation.environment_state": np.zeros(3, dtype=np.float32),
    }
    try:
        first = runtime.infer(features, "task", 1)
        runtime.acknowledge(first["action_id"])
        runtime.infer(features, "task", 2)
    finally:
        runtime.close()
    assert len(seen) == 2
    assert all(kwargs["inference_mode"] == mode for kwargs in seen)
    assert all(kwargs["tactile_cfg_lambda"] == 2.25 for kwargs in seen)


@pytest.mark.parametrize("strength", [-1.0, float("nan"), float("inf")])
def test_runtime_rejects_invalid_tactile_strength(strength):
    with pytest.raises(ValueError, match="tactile_cfg_lambda must be finite and non-negative"):
        RuntimeSession(runner_for(ConformingPolicy(tiny_config())), queue_size=8, tactile_cfg_lambda=strength)


@pytest.mark.parametrize("unit", ["tokens", "chunks"])
def test_stream_head_passes_current_prefix_age_to_policy_batch(unit):
    policy = ConformingPolicy(tiny_config())
    policy.config.streaming = True
    policy.config.streaming_chunk_size = 2
    setattr(policy.config, f"observation_delay_max_{unit}", 3)
    key = f"observation_delay_{unit}"
    seen = []
    policy.encode_prefix = lambda batch: {}

    def predict_from_prefix(batch, cache, **kwargs):
        assert kwargs["inference_mode"] == "stream-head"
        seen.append(batch[key].item())
        return policy.predict_action_chunk(batch)

    policy.predict_action_chunk_from_prefix = predict_from_prefix
    runtime = RuntimeSession(
        runner_for(policy),
        queue_size=8,
        inference_mode="stream-head",
        streaming_horizon=100,
        streaming_watermark=0,
    )
    features = {
        "observation.state": np.zeros(3, dtype=np.float32),
        "observation.environment_state": np.zeros(3, dtype=np.float32),
    }
    try:
        for step in range(9):
            action = runtime.infer(features, "task", step)
            runtime.infer(features, "task", step)  # Replays must not advance age or run FM.
            runtime.acknowledge(action["action_id"])
        runtime.infer(features, "new task", 9)  # A fresh prefix resets its age.
    finally:
        runtime.close()
    divisor = 2 if unit == "chunks" else 1
    assert seen == [min(step // divisor, 3) for step in range(9)] + [0]

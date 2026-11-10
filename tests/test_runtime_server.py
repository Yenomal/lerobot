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

import numpy as np
import pytest

from lerobot.runtime_protocol import acknowledge, action_body, observation
from lerobot.runtime_server import RuntimeSession
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


def test_runtime_protocol_round_trip_keeps_observation_arrays_and_ack_identity():
    pytest.importorskip("msgpack")
    message = observation("robot-a", 4, {"state": np.array([1, 2], dtype=np.float32)}, task="move")
    decoded = decode_message(encode_message(message))
    assert decoded.message_type is MessageType.OBSERVATION
    assert decoded.body["features"]["state"].tolist() == [1.0, 2.0]
    ack = acknowledge("robot-a", 9)
    assert action_body(
        decoded.reply(
            MessageType.ACTION,
            {"op": "action", "action_id": 9, "action": np.array([0.5], dtype=np.float32)},
        )
    )["action_id"] == 9
    with pytest.raises(ValueError):
        action_body(ack)

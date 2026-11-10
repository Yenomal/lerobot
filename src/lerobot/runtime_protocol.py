"""Small observation/action protocol for the standalone Runtime server.

The protocol intentionally knows nothing about robots, policies or LeRobot rollout.
An external client sends an observation, receives one action, and acknowledges the
action before continuing. Messages use the existing bounded NumPy/MessagePack codec
but have their own operation names and wire contract.
"""

from __future__ import annotations

import asyncio
import struct
from typing import Any
from uuid import uuid4

import numpy as np

from lerobot.transport.protocol import Envelope, MessageType, decode_message, encode_message

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 64 * 1024 * 1024


def observation(session_id: str, sequence: int, features: dict[str, np.ndarray], *, task: str = "") -> Envelope:
    """Build one client observation request."""
    return Envelope(
        MessageType.OBSERVATION,
        instance_id="runtime",
        session_id=session_id,
        request_id=uuid4().hex,
        body={"protocol_version": PROTOCOL_VERSION, "op": "observe", "sequence": sequence, "features": features, "task": task},
    )


def acknowledge(session_id: str, action_id: int) -> Envelope:
    """Build an acknowledgement for an action already received by the client."""
    return Envelope(
        MessageType.CONTROL,
        instance_id="runtime",
        session_id=session_id,
        request_id=uuid4().hex,
        body={"protocol_version": PROTOCOL_VERSION, "op": "ack", "action_id": action_id},
    )


def reset(session_id: str) -> Envelope:
    """Build a session reset request."""
    return Envelope(
        MessageType.CONTROL,
        instance_id="runtime",
        session_id=session_id,
        request_id=uuid4().hex,
        body={"protocol_version": PROTOCOL_VERSION, "op": "reset"},
    )


def _frame(payload: bytes) -> bytes:
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("Runtime message exceeds the maximum frame size")
    return struct.pack(">I", len(payload)) + payload


async def read_message(reader: asyncio.StreamReader) -> Envelope:
    """Read one length-prefixed MessagePack envelope."""
    header = await reader.readexactly(4)
    size = struct.unpack(">I", header)[0]
    if size <= 0 or size > MAX_FRAME_BYTES:
        raise ValueError("Invalid Runtime frame size")
    payload = await reader.readexactly(size)
    return decode_message(payload)


async def write_message(writer: asyncio.StreamWriter, message: Envelope) -> None:
    """Write one envelope and wait until it is handed to the socket."""
    writer.write(_frame(encode_message(message)))
    await writer.drain()


def action_body(message: Envelope) -> dict[str, Any]:
    """Validate and return a server action response body."""
    if message.message_type is not MessageType.ACTION or message.body.get("op") != "action":
        raise ValueError("Expected a Runtime action response")
    action = message.body.get("action")
    if not isinstance(action, np.ndarray) or action.ndim != 1 or action.dtype.name != "float32":
        raise ValueError("Runtime action must be a one-dimensional float32 array")
    if type(message.body.get("action_id")) is not int:
        raise ValueError("Runtime action is missing action_id")
    return message.body

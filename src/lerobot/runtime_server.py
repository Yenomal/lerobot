"""Standalone server-side Runtime for one observation/action control stream."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from huggingface_hub import snapshot_download

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.inference import ExecutionMode, FeatureSpec, ObservationSnapshot, PolicyRunner
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.runtime_protocol import PROTOCOL_VERSION, read_message, write_message
from lerobot.transport.protocol import Envelope, ErrorCode, MessageType

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    """Validate requested accelerator and explain CUDA build/driver mismatches early."""
    try:
        torch_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid torch device {device!r}") from exc
    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {device}, but CUDA is unavailable to torch {torch.__version__} "
                f"(CUDA build {torch.version.cuda!r}). Check that the container can access the GPU and "
                "that the PyTorch CUDA build is supported by the host driver. On Linux x86_64, "
                "recreate the environment with `uv sync --locked --extra remote --extra smolvla`; "
                "this project lock uses CUDA 12.6 wheels for drivers supporting CUDA 12.x."
            )
        if torch_device.index is not None and torch_device.index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible"
            )
    return str(torch_device)


def _feature_specs(config: PreTrainedConfig) -> tuple[tuple[FeatureSpec, ...], FeatureSpec]:
    if not config.input_features or not config.output_features:
        raise ValueError("Checkpoint config must contain input_features and output_features")
    features = []
    for name, feature in config.input_features.items():
        visual = feature.type is FeatureType.VISUAL
        shape = tuple(feature.shape)
        if visual:
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"Checkpoint visual feature {name!r} must use CHW shape with three channels")
            shape = (shape[1], shape[2], shape[0])
        features.append(
            FeatureSpec(
                name,
                shape,
                "uint8" if visual else "float32",
                kind="rgb" if visual else "tensor",
                semantics="checkpoint",
            )
        )
    output = config.action_feature
    if output is None:
        raise ValueError("Checkpoint config must contain an action output feature")
    action_names = tuple(getattr(config, "action_feature_names", ()) or ())
    action = FeatureSpec(
        "action",
        tuple(output.shape),
        "float32",
        names=action_names,
        semantics="checkpoint",
    )
    return tuple(features), action


def load_runtime_runner(checkpoint: str, device: str) -> PolicyRunner:
    """Load a checkpoint using only its saved model and processor configuration."""
    device = _resolve_device(device)
    path = Path(checkpoint)
    resolved = path.resolve() if path.is_dir() else Path(snapshot_download(checkpoint))
    config = PreTrainedConfig.from_pretrained(resolved)
    config.pretrained_path = resolved
    config.device = device
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(resolved, config=config)
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=resolved,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    features, action = _feature_specs(config)
    return PolicyRunner(
        policy,
        preprocessor,
        postprocessor,
        action_interval=1.0,
        features=features,
        action_feature=action,
        modes=(ExecutionMode.CHUNK,),
    )


class RuntimeSession:
    """Server-owned action queue and at-least-once action delivery."""

    def __init__(self, runner: PolicyRunner, queue_size: int) -> None:
        if queue_size <= 0:
            raise ValueError("runtime_queue_size must be positive")
        self.runner = runner
        self.queue_size = queue_size
        self.actions: deque[np.ndarray] = deque()
        self.pending: dict[str, Any] | None = None
        self.next_action_id = 0
        self.last_acknowledged_id: int | None = None
        self.task: str | None = None

    def reset(self) -> None:
        self.actions.clear()
        self.pending = None
        self.next_action_id = 0
        self.last_acknowledged_id = None
        self.task = None
        self.runner.reset(full=True)

    def acknowledge(self, action_id: int) -> None:
        if action_id == self.last_acknowledged_id:
            return
        if self.pending is None or action_id != self.pending["action_id"]:
            raise ValueError("Action acknowledgement does not match the pending action")
        self.last_acknowledged_id = action_id
        self.pending = None

    def infer(self, features: dict[str, np.ndarray], task: str, sequence: int) -> dict[str, Any]:
        if self.pending is not None:
            # The client may have lost the response. Replaying the exact action
            # is safer than advancing the queue without confirmation.
            return {**self.pending, "replayed": True, "sequence": sequence}
        if self.task is not None and task != self.task:
            # Never execute a queued plan conditioned on an earlier instruction.
            self.actions.clear()
        self.task = task
        if not self.actions:
            observation = ObservationSnapshot(features, 0.0, task, observation_id=str(sequence))
            chunk = self.runner.predict(observation, mode=ExecutionMode.CHUNK)
            actions = chunk.canonical_actions.detach().cpu().numpy().astype(np.float32, copy=True)
            if actions.ndim != 2 or len(actions) == 0:
                raise ValueError("Policy returned an empty action horizon")
            self.actions.extend(actions[1 : self.queue_size])
            action = actions[0]
        else:
            action = self.actions.popleft()
        response = {
            "op": "action",
            "protocol_version": PROTOCOL_VERSION,
            "action_id": self.next_action_id,
            "sequence": sequence,
            "action": np.asarray(action, dtype=np.float32),
            "queue_size": len(self.actions),
        }
        self.next_action_id += 1
        self.pending = response
        return response


class RuntimeServer:
    def __init__(self, runner: PolicyRunner, host: str, port: int, queue_size: int) -> None:
        self.runner = runner
        self.host = host
        self.port = port
        self.queue_size = queue_size
        self._active = False

    async def serve(self) -> None:
        server = await asyncio.start_server(self._connection, self.host, self.port)
        sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or ())
        logger.info("Runtime server listening on %s", sockets)
        # Keep a direct terminal signal after the socket bind. This remains
        # visible even when a caller's logging configuration filters INFO logs.
        print(f"Runtime server listening on {sockets}", flush=True)
        async with server:
            await server.serve_forever()

    async def _connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._active:
            await write_message(writer, Envelope(MessageType.ERROR, "runtime", "client", request_id=uuid4().hex, body={"code": str(ErrorCode.BUSY), "message": "Runtime already has an active client"}))
            writer.close()
            await writer.wait_closed()
            return
        self._active = True
        session = RuntimeSession(self.runner, self.queue_size)
        request: Envelope | None = None
        last_sequence = -1
        try:
            while True:
                request = await read_message(reader)
                if request.body.get("protocol_version") != PROTOCOL_VERSION:
                    raise ValueError("Unsupported Runtime protocol version")
                if request.message_type is MessageType.CONTROL:
                    op = request.body.get("op")
                    if op == "reset":
                        await asyncio.to_thread(session.reset)
                        last_sequence = -1
                        await write_message(writer, request.reply(MessageType.ACK, {"op": "reset"}))
                    elif op == "ack":
                        session.acknowledge(request.body.get("action_id"))
                        await write_message(writer, request.reply(MessageType.ACK, {"op": "ack"}))
                    else:
                        raise ValueError(f"Unknown Runtime control operation: {op!r}")
                    continue
                if request.message_type is not MessageType.OBSERVATION or request.body.get("op") != "observe":
                    raise ValueError("Runtime expects observation or control messages")
                features = request.body.get("features")
                task = request.body.get("task", "")
                sequence = request.body.get("sequence")
                if not isinstance(features, dict) or not isinstance(task, str) or type(sequence) is not int:
                    raise ValueError("Malformed Runtime observation")
                if sequence < last_sequence or (sequence == last_sequence and session.pending is None):
                    raise ValueError("Observation sequence must increase monotonically")
                last_sequence = max(last_sequence, sequence)
                result = await asyncio.to_thread(session.infer, features, task, sequence)
                await write_message(writer, request.reply(MessageType.ACTION, result))
        except (asyncio.IncompleteReadError, ConnectionError):
            logger.info("Runtime client disconnected")
        except Exception as exc:
            logger.exception("Runtime session failed")
            try:
                if request is not None:
                    await write_message(writer, request.error(ErrorCode.EXECUTION, str(exc)[:2048]))
            except Exception:
                pass
        finally:
            try:
                await asyncio.to_thread(self.runner.reset, full=True)
            except Exception:
                logger.exception("Runtime policy reset failed after client disconnect")
            self._active = False
            writer.close()
            await writer.wait_closed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a LeRobot policy behind a server-owned Runtime queue")
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint directory or Hub repository")
    parser.add_argument("--device", default="cuda", help="Torch device used to load the policy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--runtime-queue-size", type=int, default=64)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    device = _resolve_device(args.device)
    runner = load_runtime_runner(args.checkpoint, device)
    asyncio.run(RuntimeServer(runner, args.host, args.port, args.runtime_queue_size).serve())


if __name__ == "__main__":
    main()

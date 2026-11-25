"""Process-isolated VLM/FM workers for SmolVLA Runtime streaming."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from multiprocessing.connection import Connection
from threading import Lock
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.inference import ActionChunk, ActionProvenance, ObservationSnapshot, PolicyRunner
from lerobot.runtime_server import load_runtime_runner


def _observation_metadata(observation: ObservationSnapshot) -> dict[str, Any]:
    metadata: dict[str, Any] = {"capture_time": observation.capture_time}
    if observation.tactile_hmm_state is not None:
        metadata["tactile_hmm_state"] = observation.tactile_hmm_state
    if observation.tactile_history is not None:
        metadata.update(
            tactile_history=dict(observation.tactile_history),
            tactile_history_times=observation.tactile_history_times,
        )
    return metadata


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu").contiguous()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def _encode_cache(value: Any) -> Any:
    """Encode a DynamicCache into a pickle-safe tensor tree."""
    if value is None:
        return None
    if hasattr(value, "layers") and value.__class__.__name__ == "DynamicCache":
        layers = []
        for layer in value.layers:
            sliding_window = getattr(layer, "sliding_window", None)
            layers.append(
                {
                    "keys": _cpu_tree(layer.keys),
                    "values": _cpu_tree(layer.values),
                    "sliding_window": sliding_window,
                }
            )
        return {"__dynamic_cache__": True, "layers": layers}
    if isinstance(value, Mapping):
        return {key: _encode_cache(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_encode_cache(item) for item in value)
    if isinstance(value, list):
        return [_encode_cache(item) for item in value]
    if torch.is_tensor(value):
        return _cpu_tree(value)
    return value


def _decode_cache(value: Any, device: str) -> Any:
    if isinstance(value, dict) and value.get("__dynamic_cache__"):
        from transformers import DynamicCache

        entries = []
        for layer in value["layers"]:
            keys = layer["keys"].to(device=device)
            values = layer["values"].to(device=device)
            sliding_window = layer.get("sliding_window")
            entries.append((keys, values, sliding_window) if sliding_window is not None else (keys, values))
        return DynamicCache(ddp_cache_data=entries)
    if isinstance(value, dict):
        return {key: _decode_cache(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_decode_cache(item, device) for item in value)
    if isinstance(value, list):
        return [_decode_cache(item, device) for item in value]
    if torch.is_tensor(value):
        return value.to(device=device)
    return value


def _worker_request(
    role: str,
    runner: PolicyRunner,
    device: str,
    sessions: dict[str, PolicyRunner],
    fm_caches: dict[str, tuple[str, Any]],
    request: dict[str, Any],
) -> dict[str, Any]:
    session_id = request["session_id"]
    op = request["op"]
    if op in ("reset", "release"):
        fm_caches.pop(session_id, None)
        if op == "release":
            sessions.pop(session_id, None)
        elif session_id in sessions:
            sessions[session_id].reset(full=True)
        return {"ok": True}

    session = sessions.get(session_id)
    if session is None:
        session = sessions[session_id] = runner.for_session(session_id)
    observation = ObservationSnapshot(
        request["features"],
        request.get("capture_time", 0.0),
        request["task"],
        observation_id=str(request["sequence"]),
        tactile_history=request.get("tactile_history"),
        tactile_history_times=tuple(request.get("tactile_history_times", ())),
        tactile_hmm_state=request.get("tactile_hmm_state"),
    )
    if role == "vlm" and op == "encode_prefix":
        prepared, cache = session.encode_prefix(observation)
        return {"prepared": _cpu_tree(prepared), "cache": _encode_cache(cache)}
    if role == "fm" and op == "predict_from_prefix":
        cache_id = request["cache_id"]
        if "cache" in request:
            fm_cache = _decode_cache(request["cache"], device)
        else:
            loaded = fm_caches.get(session_id)
            if loaded is None or loaded[0] != cache_id:
                raise ValueError("FM prefix cache is not loaded; submit the cache before reusing its ID")
            _, fm_cache = loaded
        result = session.predict_from_prefix(
            {},
            fm_cache,
            observation=observation,
            inference_mode=request["inference_mode"],
            context_cfg_lambda=float(request["context_cfg_lambda"]),
            tactile_cfg_lambda=float(request["tactile_cfg_lambda"]),
            observation_delay_tokens=request.get("observation_delay_tokens"),
            observation_delay_chunks=request.get("observation_delay_chunks"),
        )
        fm_caches[session_id] = (cache_id, fm_cache)
        return {"actions": result.canonical_actions.cpu().numpy()}
    raise ValueError(f"Unsupported {role} worker operation: {op!r}")


def _worker_main(
    role: str, checkpoint: str, device: str, connection: Connection, exp_random_tactile: bool = False
) -> None:
    runner = load_runtime_runner(
        checkpoint, device, **({"exp_random_tactile": True} if exp_random_tactile else {})
    )
    sessions: dict[str, PolicyRunner] = {}
    fm_caches: dict[str, tuple[str, Any]] = {}
    try:
        connection.send({"ready": role})
        while True:
            request = connection.recv()
            if request["op"] == "close":
                connection.send({"ok": True})
                return
            try:
                response = _worker_request(role, runner, device, sessions, fm_caches, request)
            except Exception as exc:
                response = {"error": f"{type(exc).__name__}: {exc}"}
            connection.send(response)
            # Do not pin tensors sent to the parent while the worker is idle.
            del response, request
    except (EOFError, OSError):
        return
    finally:
        connection.close()


class ProcessStreamingRunner:
    """PolicyRunner-compatible parent proxy for split VLM/FM workers."""

    def __init__(self, checkpoint: str, vlm_device: str, fm_device: str, *, exp_random_tactile: bool = False):
        import multiprocessing as mp

        self.checkpoint = checkpoint
        resolved = checkpoint
        if not resolved:
            raise ValueError("checkpoint is required")
        self.config = PreTrainedConfig.from_pretrained(resolved)
        self.policy = SimpleNamespace(config=self.config)
        self._vlm_lock = Lock()
        self._fm_lock = Lock()
        self._session_id = ""
        self._fm_prefix_cache: Any = None
        self._fm_prefix_id: str | None = None
        context = mp.get_context("spawn")
        self._vlm_parent, vlm_child = context.Pipe()
        self._fm_parent, fm_child = context.Pipe()
        self._vlm_process = context.Process(
            target=_worker_main,
            args=("vlm", checkpoint, vlm_device, vlm_child),
            daemon=True,
        )
        self._fm_process = context.Process(
            target=_worker_main,
            args=("fm", checkpoint, fm_device, fm_child, exp_random_tactile),
            daemon=True,
        )
        self._vlm_process.start()
        self._fm_process.start()
        for connection, role in ((self._vlm_parent, "vlm"), (self._fm_parent, "fm")):
            try:
                ready = connection.recv()
            except (EOFError, OSError) as exc:
                raise RuntimeError(f"{role.upper()} worker exited during startup") from exc
            if ready.get("ready") != role:
                raise RuntimeError(f"Unexpected {role.upper()} worker startup response: {ready!r}")

    def for_session(self, session_id: str) -> ProcessStreamingRunner:
        """Bind a session to the existing workers with its own prefix tracking."""
        runner = copy(self)
        runner._session_id = session_id
        runner._fm_prefix_cache = None
        runner._fm_prefix_id = None
        return runner

    def encode_prefix(self, observation: ObservationSnapshot) -> tuple[dict[str, Any], Any]:
        # A background refresh and a foreground reset must not receive each
        # other's replies from the same pipe.
        with self._vlm_lock:
            self._vlm_parent.send(
                {
                    "op": "encode_prefix",
                    "session_id": self._session_id,
                    "features": dict(observation.features),
                    "task": observation.task,
                    "sequence": observation.observation_id,
                    **_observation_metadata(observation),
                }
            )
            result = self._vlm_parent.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["prepared"], result["cache"]

    def predict(self, observation: ObservationSnapshot, *, mode=None, **kwargs) -> ActionChunk:
        """Provide the ordinary full-chunk runner surface for Runtime fallback paths."""
        del mode
        prepared, cache = self.encode_prefix(observation)
        return self.predict_from_prefix(
            prepared,
            cache,
            observation=observation,
            inference_mode=kwargs.get("inference_mode", "stream"),
            context_cfg_lambda=float(kwargs.get("context_cfg_lambda", 0.1)),
            tactile_cfg_lambda=float(kwargs.get("tactile_cfg_lambda", 1.5)),
        )

    def predict_from_prefix(
        self,
        prepared: dict[str, Any],
        prefix_cache: Any,
        *,
        observation: ObservationSnapshot,
        inference_mode: str,
        context_cfg_lambda: float,
        tactile_cfg_lambda: float = 1.5,
        observation_delay_tokens: int | None = None,
        observation_delay_chunks: int | None = None,
    ) -> ActionChunk:
        # PolicyRunner prepares the current observation on the FM side. The
        # VLM's prepared batch is unused here and need not cross this pipe.
        del prepared
        new_prefix = self._fm_prefix_id is None or prefix_cache is not self._fm_prefix_cache
        cache_id = uuid4().hex if new_prefix else self._fm_prefix_id
        request = {
            "op": "predict_from_prefix",
            "session_id": self._session_id,
            "cache_id": cache_id,
            "features": dict(observation.features),
            "task": observation.task,
            "sequence": observation.observation_id,
            **_observation_metadata(observation),
            "inference_mode": inference_mode,
            "context_cfg_lambda": context_cfg_lambda,
            "tactile_cfg_lambda": tactile_cfg_lambda,
            "observation_delay_tokens": observation_delay_tokens,
            "observation_delay_chunks": observation_delay_chunks,
        }
        if new_prefix:
            request["cache"] = prefix_cache
        with self._fm_lock:
            self._fm_parent.send(request)
            result = self._fm_parent.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        self._fm_prefix_cache = prefix_cache
        self._fm_prefix_id = cache_id
        actions = torch.from_numpy(np.asarray(result["actions"], dtype=np.float32))
        return ActionChunk(
            model_actions=None,
            canonical_actions=actions,
            provenance=ActionProvenance(
                capture_time=observation.capture_time,
                task=observation.task,
                observation_id=observation.observation_id,
            ),
            execution_steps=len(actions),
            server_durations={},
        )

    def _session_control(self, op: str) -> None:
        self._fm_prefix_cache = None
        self._fm_prefix_id = None
        errors: list[Exception] = []
        for connection, lock in ((self._vlm_parent, self._vlm_lock), (self._fm_parent, self._fm_lock)):
            try:
                with lock:
                    connection.send({"op": op, "session_id": self._session_id})
                    response = connection.recv()
                if "error" in response:
                    raise RuntimeError(response["error"])
            except (EOFError, OSError, RuntimeError) as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def reset(self, *, full: bool = True) -> None:
        """Reset this session on both workers."""
        self._session_control("reset")

    def release(self) -> None:
        """Release this session's worker allocations after disconnect."""
        self._session_control("release")

    def close(self) -> None:
        """Shut down both shared workers when the server stops."""
        for connection, lock in ((self._vlm_parent, self._vlm_lock), (self._fm_parent, self._fm_lock)):
            with lock:
                try:
                    connection.send({"op": "close"})
                    connection.recv()
                except (EOFError, OSError):
                    pass
                finally:
                    connection.close()
        for process in (self._vlm_process, self._fm_process):
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)

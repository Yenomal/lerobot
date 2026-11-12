"""Process-isolated VLM/FM workers for SmolVLA Runtime streaming."""

from __future__ import annotations

from multiprocessing.connection import Connection
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.inference import ActionChunk, ActionProvenance, ObservationSnapshot, PolicyRunner
from lerobot.runtime_server import load_runtime_runner


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


def _worker_main(role: str, checkpoint: str, device: str, connection: Connection) -> None:
    runner = load_runtime_runner(checkpoint, device)
    try:
        connection.send({"ready": role})
        while True:
            request = connection.recv()
            op = request.get("op")
            if op == "close":
                connection.send({"ok": True})
                return
            if op == "reset":
                runner.reset(full=True)
                connection.send({"ok": True})
                continue
            if role == "vlm" and op == "encode_prefix":
                observation = ObservationSnapshot(
                    request["features"],
                    0.0,
                    request["task"],
                    observation_id=str(request["sequence"]),
                )
                prepared, cache = runner.encode_prefix(observation)
                connection.send({"prepared": _cpu_tree(prepared), "cache": _encode_cache(cache)})
                continue
            if role == "fm" and op == "predict_from_prefix":
                cache = _decode_cache(request["cache"], device)
                prepared = _decode_cache(request["prepared"], device)
                observation = ObservationSnapshot(
                    request["features"],
                    0.0,
                    request["task"],
                    observation_id=str(request["sequence"]),
                )
                result = runner.predict_from_prefix(
                    prepared,
                    cache,
                    observation=observation,
                    inference_mode=request["inference_mode"],
                    context_cfg_lambda=float(request["context_cfg_lambda"]),
                    observation_delay_tokens=request.get("observation_delay_tokens"),
                    observation_delay_chunks=request.get("observation_delay_chunks"),
                )
                connection.send({"actions": result.canonical_actions.cpu().numpy()})
                continue
            raise ValueError(f"Unsupported {role} worker operation: {op!r}")
    except BaseException as exc:
        try:
            connection.send({"error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
        raise
    finally:
        connection.close()


class ProcessStreamingRunner:
    """PolicyRunner-compatible parent proxy for split VLM/FM workers."""

    def __init__(self, checkpoint: str, vlm_device: str, fm_device: str):
        import multiprocessing as mp

        self.checkpoint = checkpoint
        resolved = checkpoint
        if not resolved:
            raise ValueError("checkpoint is required")
        self.config = PreTrainedConfig.from_pretrained(resolved)
        self.policy = SimpleNamespace(config=self.config)
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
            args=("fm", checkpoint, fm_device, fm_child),
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

    def encode_prefix(self, observation: ObservationSnapshot):
        self._vlm_parent.send(
            {
                "op": "encode_prefix",
                "features": dict(observation.features),
                "task": observation.task,
                "sequence": observation.observation_id,
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
        )

    def predict_from_prefix(
        self,
        prepared: dict[str, Any],
        prefix_cache: Any,
        *,
        observation: ObservationSnapshot,
        inference_mode: str,
        context_cfg_lambda: float,
        observation_delay_tokens: int | None = None,
        observation_delay_chunks: int | None = None,
    ) -> ActionChunk:
        self._fm_parent.send(
            {
                "op": "predict_from_prefix",
                "prepared": prepared,
                "cache": prefix_cache,
                "features": dict(observation.features),
                "task": observation.task,
                "sequence": observation.observation_id,
                "inference_mode": inference_mode,
                "context_cfg_lambda": context_cfg_lambda,
                "observation_delay_tokens": observation_delay_tokens,
                "observation_delay_chunks": observation_delay_chunks,
            }
        )
        result = self._fm_parent.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
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

    def reset(self, *, full: bool = True) -> None:
        del full
        for connection in (self._vlm_parent, self._fm_parent):
            try:
                connection.send({"op": "reset"})
            except (BrokenPipeError, EOFError, OSError):
                continue
        for connection in (self._vlm_parent, self._fm_parent):
            try:
                connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                continue

    def close(self) -> None:
        for connection in (self._vlm_parent, self._fm_parent):
            try:
                connection.send({"op": "close"})
                connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in (self._vlm_process, self._fm_process):
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)

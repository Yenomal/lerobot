"""Frozen two-sided PCA and causal Gaussian HMM filtering for UniVTac markers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata


TACTILE_HMM_STATE = "observation.tactile.hmm_state"
TACTILE_HMM_RAW = "observation.tactile.hmm_raw"
TACTILE_HMM_STATES = 8
MARKER_SHAPE = (2, 63, 2)


class TactileHMM:
    """Read the fixed PCA bases and Gaussian HMM from dataset metadata."""

    def __init__(self, metadata: dict):
        if (
            metadata["schema_version"] != 1
            or tuple(metadata["marker_shape"]) != MARKER_SHAPE
            or metadata["displacement"] != "current_minus_reference"
        ):
            raise ValueError("Unsupported tactile HMM metadata schema or marker layout")
        self.features = tuple(metadata["features"])
        self.policy_features = tuple(metadata.get("policy_features", self.features))
        if (
            len(self.features) != 2
            or len(set(self.features)) != 2
            or len(self.policy_features) != 2
            or len(set(self.policy_features)) != 2
        ):
            raise ValueError("Tactile HMM requires two distinct marker features")
        self.pca_mean, self.pca_components = (
            np.array([side[key] for side in metadata["pca"]], dtype=np.float64)
            for key in ("mean", "components")
        )
        hmm = metadata["hmm"]
        start, transition, self.means, covariance = (
            np.array(hmm[key], dtype=np.float64) for key in ("startprob", "transmat", "means", "covars")
        )
        states = TACTILE_HMM_STATES
        arrays = [
            (self.pca_mean, (2, 126)),
            (self.pca_components, (2, 15, 126)),
            (start, (states,)),
            (transition, (states, states)),
            (self.means, (states, 30)),
            (covariance, (states, 30, 30)),
        ]
        if any(array.shape != shape or not np.isfinite(array).all() for array, shape in arrays):
            raise ValueError("Expected two 15-component PCAs and an eight-state, 30-dimensional HMM")
        if not np.allclose(covariance, covariance.swapaxes(-1, -2)):
            raise ValueError("Tactile HMM covariances must be symmetric")
        if (
            np.any(start < 0)
            or np.any(transition < 0)
            or not np.isclose(start.sum(), 1)
            or not np.allclose(transition.sum(axis=1), 1)
        ):
            raise ValueError("Invalid tactile HMM initial or transition probabilities")
        with np.errstate(divide="ignore"):
            self.log_start = np.log(start)
            self.log_transition = np.log(transition)
        cholesky = np.linalg.cholesky(covariance)
        self.whitener = np.linalg.solve(cholesky, np.broadcast_to(np.eye(30), cholesky.shape))
        self.log_scale = np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)).sum(axis=-1)

    def log_emissions(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Evaluate each state's Gaussian for current-minus-reference, left then right."""
        projections = []
        for side, value in enumerate((left, right)):
            value = np.asarray(value, dtype=np.float64)
            if value.shape[-3:] != MARKER_SHAPE or not np.isfinite(value).all():
                raise ValueError("Expected finite tactile markers with shape [..., 2, 63, 2]")
            delta = (value[..., 1, :, :] - value[..., 0, :, :]).reshape(*value.shape[:-3], 126)
            centered = delta - self.pca_mean[side]
            projections.append(np.einsum("...d,kd->...k", centered, self.pca_components[side]))
        observation = np.concatenate(projections, axis=-1)
        centered = observation[..., None, :] - self.means
        whitened = np.einsum("kij,...kj->...ki", self.whitener, centered)
        return -0.5 * np.sum(whitened**2, axis=-1) - self.log_scale

    def update(self, log_emission: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        """Return normalized log p(state_t | observations through t)."""
        prediction = (
            self.log_start
            if previous is None
            else np.logaddexp.reduce(previous[..., :, None] + self.log_transition, axis=-2)
        )
        posterior = prediction + log_emission
        return posterior - np.logaddexp.reduce(posterior, axis=-1, keepdims=True)


class DatasetTactileHMM:
    """Resolve shuffled batch rows by episode and filter each episode lazily in RAM."""

    def __init__(self, metadata: LeRobotDatasetMetadata, model: TactileHMM):
        self.metadata = metadata
        self.model = model
        self.states: dict[int, np.ndarray] = {}

    def _episode(self, episode_index: int) -> np.ndarray:
        import pyarrow.parquet as pq

        path = self.metadata.root / self.metadata.get_data_file_path(episode_index)
        table = pq.read_table(
            path,
            columns=["frame_index", *self.model.features],
            filters=[("episode_index", "=", episode_index)],
        ).sort_by("frame_index")
        frames = table["frame_index"].to_numpy()
        if not len(frames) or not np.array_equal(frames, np.arange(len(frames))):
            raise ValueError(f"Missing or non-contiguous tactile frames for episode {episode_index}")
        values = [
            table[name].combine_chunks().flatten().flatten().flatten().to_numpy().reshape(-1, *MARKER_SHAPE)
            for name in self.model.features
        ]
        emissions = self.model.log_emissions(*values)
        states = np.empty(len(frames), dtype=np.int64)
        posterior = None
        for index, emission in enumerate(emissions):
            posterior = self.model.update(emission, posterior)
            states[index] = posterior.argmax()
        return states

    def __call__(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        import torch

        ids = []
        for episode, frame in zip(
            batch["episode_index"].tolist(), batch["frame_index"].tolist(), strict=True
        ):
            if episode not in self.states:
                self.states[episode] = self._episode(episode)
            sequence = self.states[episode]
            if not 0 <= frame < len(sequence):
                raise ValueError(f"Frame {frame} is outside episode {episode}")
            ids.append(sequence[frame])
        return torch.tensor(ids, dtype=torch.long, device=batch["frame_index"].device)

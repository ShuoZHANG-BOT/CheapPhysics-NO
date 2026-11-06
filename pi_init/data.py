"""Loading PDEBench HDF5 files into FNO tensors.

Trajectories use the PDEBench FNO layout ``[sample, *space, time, channel]``.
Cheap inputs (initial conditions or coefficient fields) use ``[sample, *space, channel]``
and are read from the file without touching any later time step or solution field.

Supported layouts
-----------------
* ``scalar1d``: ``tensor[N, T, X]`` (Burgers, Advection, 1D Reaction-Diffusion).
* ``cfd1d``:    ``density/pressure/Vx[N, T, X]`` (1D compressible Navier-Stokes).
* ``darcy``:    ``nu[N, X, Y]`` coefficient and ``tensor[N, 1, X, Y]`` solution.
  As in PDEBench, the coefficient and solution are stacked as two "time" frames,
  so the operator maps frame 0 (``nu``) to frame 1 (``u``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch

CFD_CHANNELS = ("density", "pressure", "Vx")


@dataclass
class Split:
    """Indices into the training file and where the test samples come from."""

    pool: np.ndarray          # candidate pool for labels and cheap inputs
    test_file: Path
    test_indices: np.ndarray


@dataclass
class Grid:
    coords: dict              # reduced coordinate arrays, e.g. {"x": ..., "t": ...}
    grid: torch.Tensor        # [*space, dim], model input coordinates
    attrs: dict = field(default_factory=dict)


def _read_rows(dataset: h5py.Dataset, indices: np.ndarray, tail: tuple, chunk: int = 512) -> np.ndarray:
    """Read ``dataset[indices, *tail]`` in sorted chunks and restore the requested order."""
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    sorted_idx = indices[order]
    parts = [
        np.asarray(dataset[(sorted_idx[i:i + chunk].tolist(),) + tuple(tail)], dtype=np.float32)
        for i in range(0, len(sorted_idx), chunk)
    ]
    values = np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)
    restored = np.empty_like(values)
    restored[order] = values
    return restored


def split(cfg: dict) -> Split:
    """PDEBench convention: the first ``test_ratio`` of a single file is the test set."""
    data = cfg["data"]
    train_file = Path(data["root"]) / data["train_file"]
    with h5py.File(train_file, "r") as handle:
        key = "nu" if data["layout"] == "darcy" else ("density" if data["layout"] == "cfd1d" else "tensor")
        total = int(handle[key].shape[0])
    if data.get("test_file"):
        test_file = Path(data["root"]) / data["test_file"]
        with h5py.File(test_file, "r") as handle:
            test_total = int(handle["tensor"].shape[0])
        return Split(np.arange(total), test_file, np.arange(test_total))
    n_test = int(total * data.get("test_ratio", 0.1))
    return Split(np.arange(n_test, total), train_file, np.arange(n_test))


def load_grid(cfg: dict, path: Path | None = None) -> Grid:
    data = cfg["data"]
    path = path or Path(data["root"]) / data["train_file"]
    rx, rt = data["reduced_resolution"], data.get("reduced_resolution_t", 1)
    with h5py.File(path, "r") as handle:
        attrs = {k: (float(v) if np.ndim(v) == 0 and np.issubdtype(np.asarray(v).dtype, np.number) else v)
                 for k, v in handle.attrs.items()}
        x = np.asarray(handle["x-coordinate"][::rx], dtype=np.float32)
        if data["layout"] == "darcy":
            y = np.asarray(handle["y-coordinate"][::rx], dtype=np.float32)
            xx, yy = np.meshgrid(x, y, indexing="ij")
            grid = torch.from_numpy(np.stack((xx, yy), axis=-1))
            return Grid({"x": x, "y": y}, grid, attrs)
        t = np.asarray(handle["t-coordinate"][::rt], dtype=np.float32)
    return Grid({"x": x, "t": t}, torch.from_numpy(x).unsqueeze(-1), attrs)


def load_trajectories(cfg: dict, path: Path, indices: np.ndarray) -> torch.Tensor:
    """Full input-solution pairs, truncated to the ``t_train`` frames that are used."""
    data = cfg["data"]
    rx, rt, frames = data["reduced_resolution"], data.get("reduced_resolution_t", 1), data["t_train"]
    with h5py.File(path, "r") as handle:
        if data["layout"] == "scalar1d":
            values = _read_rows(handle["tensor"], indices, (slice(None, None, rt), slice(None, None, rx)))
            values = values[:, :frames].transpose(0, 2, 1)[..., None]
        elif data["layout"] == "cfd1d":
            values = np.stack([
                _read_rows(handle[k], indices, (slice(None, None, rt), slice(None, None, rx)))[:, :frames]
                for k in CFD_CHANNELS
            ], axis=-1).transpose(0, 2, 1, 3)
        elif data["layout"] == "darcy":
            coef = _read_rows(handle["nu"], indices, (slice(None, None, rx), slice(None, None, rx)))
            sol = _read_rows(handle["tensor"], indices, (0, slice(None, None, rx), slice(None, None, rx)))
            values = np.stack((coef, sol), axis=-1)[..., None]
        else:
            raise ValueError(f"unknown layout {data['layout']}")
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


def load_cheap_inputs(cfg: dict, path: Path, indices: np.ndarray) -> torch.Tensor:
    """Initial conditions (time-dependent PDEs) or coefficient fields (Darcy) only."""
    data = cfg["data"]
    rx = data["reduced_resolution"]
    with h5py.File(path, "r") as handle:
        if data["layout"] == "scalar1d":
            values = _read_rows(handle["tensor"], indices, (0, slice(None, None, rx)))[..., None]
        elif data["layout"] == "cfd1d":
            values = np.stack([_read_rows(handle[k], indices, (0, slice(None, None, rx))) for k in CFD_CHANNELS], -1)
        elif data["layout"] == "darcy":
            values = _read_rows(handle["nu"], indices, (slice(None, None, rx), slice(None, None, rx)))[..., None]
        else:
            raise ValueError(f"unknown layout {data['layout']}")
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


def sample_indices(pool: np.ndarray, count: int | None, seed: int) -> np.ndarray:
    """Uniform random subset of the pool; ``None`` or ``count >= len(pool)`` returns the whole pool."""
    if count is None or count >= len(pool):
        return np.asarray(pool, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(pool, size=count, replace=False)).astype(np.int64)

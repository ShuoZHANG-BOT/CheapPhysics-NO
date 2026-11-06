"""CPU tests: analytic residual checks and an end-to-end run on tiny synthetic files."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from pi_init.physics import advection_residual, burgers_residual, darcy_residual, euler_residual

ROOT = Path(__file__).resolve().parents[1]


def test_advection_residual_vanishes_on_exact_solution():
    x = np.linspace(0, 1, 256, endpoint=False)
    t = np.linspace(0, 0.5, 51)
    beta = 1.0
    u = np.sin(2 * math.pi * (x[:, None] - beta * t[None, :]))
    residual = advection_residual(torch.tensor(u)[None], x[1] - x[0], t[1] - t[0], beta)
    scale = 2 * math.pi
    assert residual.abs().max() / scale < 0.05


def test_burgers_residual_constant_state_is_zero():
    x = np.linspace(0, 1, 64, endpoint=False)
    t = np.linspace(0, 1, 11)
    u = torch.full((2, 64, 11), 0.7, dtype=torch.float64)
    assert burgers_residual(u, x, t, 0.01 / math.pi).abs().max() < 1e-10


def test_burgers_residual_heat_limit():
    # With a zero-mean decaying Fourier mode and the nonlinear term small, the residual
    # is dominated by the time discretization error of u_t - nu u_xx = 0.
    x = np.linspace(0, 1, 128, endpoint=False)
    t = np.linspace(0, 0.1, 201)
    nu, k, amp = 0.1, 2 * math.pi, 1e-4
    u = amp * np.exp(-nu * k ** 2 * t[None, :]) * np.sin(k * x[:, None])
    residual = burgers_residual(torch.tensor(u)[None], x, t, nu)
    assert residual.abs().max() < 1e-3 * amp * nu * k ** 2 + 1e-9


def test_euler_residual_constant_state_is_zero():
    state = torch.tensor([1.0, 0.5, 0.2]).expand(1, 32, 5, 3).clone()
    assert euler_residual(state, 0.1, 0.01, 5 / 3).abs().max() < 1e-6


def test_darcy_residual_exact_for_quadratic():
    n, beta = 33, 1.0
    x = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    u = beta / 4 * (xx * (1 - xx) + yy * (1 - yy))
    coef = torch.ones(1, n, n, dtype=torch.float64)
    residual = darcy_residual(coef, torch.tensor(u)[None], x[1] - x[0], x[1] - x[0], beta)
    assert residual.abs().max() < 1e-8


def _write_scalar(path: Path, samples: int, nt: int, nx: int, **attrs):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as handle:
        handle["tensor"] = rng.standard_normal((samples, nt, nx)).astype(np.float32)
        handle["x-coordinate"] = np.linspace(0, 1, nx, endpoint=False, dtype=np.float32)
        handle["t-coordinate"] = np.linspace(0, 1, nt + 1, dtype=np.float32)
        handle.attrs.update(attrs)


def _write_cfd(path: Path, samples: int, nt: int, nx: int):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as handle:
        for key in ("density", "pressure", "Vx"):
            handle[key] = (1.0 + 0.1 * rng.standard_normal((samples, nt, nx))).astype(np.float32)
        handle["x-coordinate"] = np.linspace(0, 1, nx, endpoint=False, dtype=np.float32)
        handle["t-coordinate"] = np.linspace(0, 1, nt + 1, dtype=np.float32)
        handle.attrs.update(eta=1e-3, zeta=1e-3)


def _write_darcy(path: Path, samples: int, n: int):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as handle:
        handle["nu"] = (1.0 + rng.random((samples, n, n))).astype(np.float32)
        handle["tensor"] = rng.random((samples, 1, n, n)).astype(np.float32)
        handle["x-coordinate"] = np.linspace(0, 1, n, dtype=np.float32)
        handle["y-coordinate"] = np.linspace(0, 1, n, dtype=np.float32)
        handle.attrs.update(beta=1.0)


@pytest.mark.parametrize("dataset", ["burgers", "advection", "cfd1d", "darcy"])
def test_end_to_end_tiny(tmp_path: Path, dataset: str):
    cfg = yaml.safe_load((ROOT / "configs" / f"{dataset}.yaml").read_text())
    data = tmp_path / "data"
    data.mkdir()
    cfg["data"]["root"] = str(data)
    if dataset == "burgers":
        _write_scalar(data / "train.hdf5", 24, 21, 128, Nu=0.01)
        _write_scalar(data / "test.hdf5", 6, 21, 128, Nu=0.01)
        cfg["data"]["t_train"] = 5
    elif dataset == "advection":
        _write_scalar(data / cfg["data"]["train_file"], 30, 21, 128, beta=1.0)
    elif dataset == "cfd1d":
        _write_cfd(data / cfg["data"]["train_file"], 30, 21, 256)
    else:
        _write_darcy(data / cfg["data"]["train_file"], 30, 64)
    cfg["pi"].update(epochs=1, batch_size=8)
    cfg["supervised"].update(epochs=2, batch_size=8, eval_every=1)
    config = tmp_path / f"{dataset}.yaml"
    config.write_text(yaml.safe_dump(cfg))
    for method in ("supervised", "pi_init"):
        subprocess.run([sys.executable, str(ROOT / "run.py"), "--config", str(config), "--method", method,
                        "--seed", "0", "--budget", "10", "--out", str(tmp_path / "runs"), "--device", "cpu"],
                       check=True, cwd=ROOT)
    results = {m: json.loads((tmp_path / "runs" / dataset / m / "B10" / "seed0" / "result.json").read_text())
               for m in ("supervised", "pi_init")}
    assert results["supervised"]["labeled_indices_sha256"] == results["pi_init"]["labeled_indices_sha256"]
    assert all(math.isfinite(r["test_relative_l2"]) for r in results.values())
    assert "pi_only_test_relative_l2" in results["pi_init"]

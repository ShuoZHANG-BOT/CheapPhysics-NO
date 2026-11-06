"""Physics-informed pretraining, supervised fine-tuning, and evaluation."""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .physics import PhysicsLoss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _batched_grid(grid: torch.Tensor, batch: int) -> torch.Tensor:
    return grid.unsqueeze(0).expand(batch, *grid.shape)


def _autoregressive(model: nn.Module, trajectory: torch.Tensor, grid: torch.Tensor,
                    initial_step: int, loss_fn=None):
    """Roll out from the first ``initial_step`` frames of ``trajectory`` ``[batch, *space, T, C]``.

    Returns the predicted trajectory and, if ``loss_fn`` is given, the sum over time
    steps of the per-step loss (the PDEBench autoregressive objective).
    """
    window = trajectory[..., :initial_step, :]
    shape = list(window.shape[:-2]) + [-1]
    predictions, loss = [window], 0.0
    for step in range(initial_step, trajectory.shape[-2]):
        prediction = model(window.reshape(shape), grid)
        if loss_fn is not None:
            target = trajectory[..., step:step + 1, :]
            loss = loss + loss_fn(prediction.reshape(len(prediction), -1), target.reshape(len(target), -1))
        predictions.append(prediction)
        window = torch.cat((window[..., 1:, :], prediction), dim=-2)
    return torch.cat(predictions, dim=-2), loss


def relative_l2(prediction: torch.Tensor, target: torch.Tensor, initial_step: int) -> torch.Tensor:
    """Per-sample relative L2 error over all predicted frames, space, and channels."""
    difference = (prediction[..., initial_step:, :] - target[..., initial_step:, :]).flatten(1)
    norm = target[..., initial_step:, :].flatten(1).norm(dim=1).clamp_min(1e-8)
    return difference.norm(dim=1) / norm


@torch.no_grad()
def evaluate(model: nn.Module, test: torch.Tensor, grid: torch.Tensor, device: torch.device,
             initial_step: int = 1, batch_size: int = 256) -> float:
    """Mean per-sample relative L2 error of an autoregressive rollout on the test set."""
    model.eval()
    grid = grid.to(device)
    errors = []
    for start in range(0, len(test), batch_size):
        trajectory = test[start:start + batch_size].to(device)
        prediction, _ = _autoregressive(model, trajectory, _batched_grid(grid, len(trajectory)), initial_step)
        errors.append(relative_l2(prediction, trajectory, initial_step).cpu())
    return float(torch.cat(errors).mean())


def pretrain(model: nn.Module, physics: PhysicsLoss, cheap_inputs: torch.Tensor, cfg: dict,
             seed: int, device: torch.device, out_dir: Path) -> list[dict]:
    """Stage 1: minimize the physics loss on cheap inputs only (no solution labels)."""
    pi = cfg["pi"]
    loader = DataLoader(TensorDataset(cheap_inputs), batch_size=pi["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=pi["lr"], weight_decay=pi.get("weight_decay", 1e-4))
    clip = pi.get("grad_clip")
    history = []
    for epoch in range(pi["epochs"]):
        model.train()
        started, total, seen = time.perf_counter(), 0.0, 0
        for (a,) in loader:
            a = a.to(device)
            loss = physics(model, a)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite physics loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total += float(loss.item()) * len(a)
            seen += len(a)
        history.append({"epoch": epoch, "pi_loss": total / seen, "seconds": time.perf_counter() - started})
        print(f"[PI] epoch {epoch + 1:4d}/{pi['epochs']} loss {history[-1]['pi_loss']:.4e}", flush=True)
    _write_csv(out_dir / "pi_history.csv", history)
    return history


def finetune(model: nn.Module, train: torch.Tensor, test: torch.Tensor, grid: torch.Tensor, cfg: dict,
             device: torch.device, out_dir: Path, seed: int) -> list[dict]:
    """Stage 2 (and the supervised-only baseline): supervised loss only, fresh Adam + StepLR."""
    sup = cfg["supervised"]
    initial_step = cfg["data"].get("initial_step", 1)
    loader = DataLoader(TensorDataset(train), batch_size=sup["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=sup["lr"], weight_decay=sup.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sup["scheduler_step"],
                                                gamma=sup["scheduler_gamma"])
    loss_fn = nn.MSELoss(reduction="mean")
    grid_device = grid.to(device)
    eval_every = sup.get("eval_every", 1)
    history = []
    for epoch in range(sup["epochs"]):
        model.train()
        started, total_loss, train_error, seen = time.perf_counter(), 0.0, 0.0, 0
        for (trajectory,) in loader:
            trajectory = trajectory.to(device)
            prediction, loss = _autoregressive(model, trajectory, _batched_grid(grid_device, len(trajectory)),
                                               initial_step, loss_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            train_error += float(relative_l2(prediction.detach(), trajectory, initial_step).sum())
            seen += len(trajectory)
        row = {"epoch": epoch, "train_loss": total_loss / len(loader), "train_relative_l2": train_error / seen,
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - started}
        last = epoch == sup["epochs"] - 1
        if last or (epoch + 1) % eval_every == 0:
            row["test_relative_l2"] = evaluate(model, test, grid, device, initial_step)
            print(f"[FT] epoch {epoch + 1:4d}/{sup['epochs']} train {row['train_relative_l2']:.4f} "
                  f"test {row['test_relative_l2']:.4f}", flush=True)
        history.append(row)
        scheduler.step()
    _write_csv(out_dir / "history.csv", [{**r, "test_relative_l2": r.get("test_relative_l2", "")} for r in history])
    return history

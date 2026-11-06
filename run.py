"""Train an FNO with supervised-only training or Physics-Informed Initialization (PI-Init).

Examples
--------
Supervised-only baseline::

    python run.py --config configs/burgers.yaml --method supervised --seed 0

PI-Init (physics pretraining on cheap inputs -> retain weights -> supervised fine-tuning)::

    python run.py --config configs/burgers.yaml --method pi_init --seed 0

Both methods use the same labeled subset, model, optimizer settings, and minibatch
order; they differ only in the parameters that supervised training starts from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from pi_init.data import load_cheap_inputs, load_grid, load_trajectories, sample_indices, split
from pi_init.fno import build_fno
from pi_init.physics import build_physics_loss
from pi_init.train import evaluate, finetune, pretrain, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--method", choices=("supervised", "pi_init"), required=True)
    p.add_argument("--seed", type=int, default=0, help="training seed (model init and minibatch order)")
    p.add_argument("--data-root", type=Path, help="override data.root in the config")
    p.add_argument("--budget", type=int, help="number of labeled trajectories (default: whole training pool)")
    p.add_argument("--n-cheap", type=int, help="number of cheap inputs for PI (default: whole training pool)")
    p.add_argument("--extra-cheap", type=Path,
                   help="optional .npy of additional solver-free inputs at the reduced resolution, "
                        "shape [M, *space] or [M, *space, channels]")
    p.add_argument("--pi-epochs", type=int, help="override pi.epochs")
    p.add_argument("--epochs", type=int, help="override supervised.epochs")
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.data_root:
        cfg["data"]["root"] = str(args.data_root)
    if args.budget is not None:
        cfg["labels"]["budget"] = args.budget
    if args.pi_epochs is not None:
        cfg["pi"]["epochs"] = args.pi_epochs
    if args.epochs is not None:
        cfg["supervised"]["epochs"] = args.epochs
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ data
    parts = split(cfg)
    train_file = Path(cfg["data"]["root"]) / cfg["data"]["train_file"]
    labeled = sample_indices(parts.pool, cfg["labels"].get("budget"), cfg["labels"].get("seed", 0))
    grid = load_grid(cfg)
    train = load_trajectories(cfg, train_file, labeled)
    test = load_trajectories(cfg, parts.test_file, parts.test_indices)
    budget = len(labeled)
    run_dir = args.out / cfg["name"] / args.method / f"B{budget}" / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"{cfg['name']}: labeled={budget} test={len(test)} trajectory shape={tuple(train.shape[1:])}", flush=True)

    # ----------------------------------------------------------------- model
    seed_everything(args.seed)
    spatial_dims = train.dim() - 3
    model = build_fno(spatial_dims, cfg["data"]["channels"], cfg["model"]["modes"], cfg["model"]["width"],
                      cfg["data"].get("initial_step", 1)).to(device)

    result = {"dataset": cfg["name"], "method": args.method, "seed": args.seed, "budget": budget,
              "labeled_indices_sha256": sha256(labeled), "test_size": len(test)}
    started = time.perf_counter()

    # --------------------------------------------- stage 1: label-free physics
    if args.method == "pi_init":
        cheap_idx = sample_indices(parts.pool, args.n_cheap, cfg["pi"].get("input_seed", 0))
        cheap = load_cheap_inputs(cfg, train_file, cheap_idx)
        if args.extra_cheap is not None:
            extra = torch.from_numpy(np.load(args.extra_cheap).astype(np.float32))
            extra = extra if extra.dim() == cheap.dim() else extra.unsqueeze(-1)
            cheap = torch.cat((cheap, extra))
        physics = build_physics_loss(cfg, grid, cheap, device)
        pi_history = pretrain(model, physics, cheap, cfg, args.seed, device, run_dir)
        pi_only = evaluate(model, test, grid.grid, device, cfg["data"].get("initial_step", 1))
        torch.save({"model_state_dict": model.state_dict()}, run_dir / "pi_checkpoint.pt")
        print(f"[PI] test relative L2 before fine-tuning (PI-only): {pi_only:.4f}", flush=True)
        result.update({"n_cheap": len(cheap), "pi_epochs": cfg["pi"]["epochs"],
                       "pi_loss_final": pi_history[-1]["pi_loss"], "pi_only_test_relative_l2": pi_only,
                       "pi_seconds": sum(r["seconds"] for r in pi_history)})

    # --------------------- stage 2: supervised loss only, fresh optimizer
    # Re-seeding makes the minibatch order identical for both methods.
    seed_everything(args.seed)
    history = finetune(model, train, test, grid.grid, cfg, device, run_dir, args.seed)
    result.update({"epochs": cfg["supervised"]["epochs"], "test_relative_l2": history[-1]["test_relative_l2"],
                   "train_relative_l2": history[-1]["train_relative_l2"],
                   "total_seconds": time.perf_counter() - started, "config": cfg})
    torch.save({"model_state_dict": model.state_dict()}, run_dir / "model.pt")
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"final test relative L2 = {result['test_relative_l2']:.4f} -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()

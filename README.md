# CheapPhysics-NO: Physics-Informed Initialization for Neural Operators

Code for **"Rethinking How to Leverage Cheap Physics for Data-Efficient Neural Operators"**.

Neural operators need input–solution pairs, and each solution comes from an expensive
numerical simulation. The PDE inputs (initial conditions, coefficient fields) and the
governing equations, by contrast, are cheap. **Physics-Informed Initialization (PI-Init)**
spends this cheap information on the *learner* rather than on choosing which inputs to label:

```text
cheap PDE inputs a (no solutions)        labeled pairs (a_i, u_i), random subset, fixed
            │                                          │
            ▼                                          │
 Stage 1  physics-informed pretraining                 │
          minimize PDE residual (+ IC/BC) of the FNO   │
          rollout on cheap inputs                       │
            │                                          │
            ▼  retain all weights, reset optimizer       ▼
 Stage 2  supervised fine-tuning with the supervised loss only
            │
            ▼
       neural operator (same architecture and inference cost as the baseline)
```

The supervised-only baseline uses **the same labeled subset, model, optimizer, schedule, and
minibatch order**; the two methods differ only in the parameters that supervised training starts from.

This repository contains a compact implementation with the FNO backbone on four representative
PDEBench problems:

| Config | Problem | Cheap input | Pretraining residual | Physics fidelity |
|---|---|---|---|---|
| `burgers` | 1D viscous Burgers, ν = 0.01/π | initial condition | u_t + u u_x − ν u_xx, periodic BC | exact |
| `advection` | 1D advection, β = 1 | initial condition | u_t + β u_x, periodic BC | exact |
| `cfd1d` | 1D compressible Navier–Stokes | initial (ρ, p, v) | inviscid Euler conservation, periodic BC | approximate |
| `darcy` | 2D Darcy flow, β = 1 | coefficient field ν | −∇·(ν∇u) = β, u = 0 on the boundary | exact |

## Installation

Python ≥ 3.10 and PyTorch ≥ 2.0. Install the PyTorch build for your hardware first, then

```bash
git clone https://github.com/ShuoZHANG-BOT/CheapPhysics-NO.git
cd CheapPhysics-NO
pip install -r requirements.txt
```

## Data

The loaders read the standard [PDEBench](https://github.com/pdebench/PDEBench) HDF5 files
(download them with PDEBench's `download_direct.py` / DaRUS links). Set `data.root` in the
config or pass `--data-root`:

| Config | File(s) | Split |
|---|---|---|
| `advection` | `1D_Advection_Sols_beta1.0.hdf5` | first 10% test, rest training pool (PDEBench convention) |
| `cfd1d` | `1D_CFD_Rand_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5` | first 10% test, rest training pool |
| `darcy` | `2D_DarcyFlow_beta1.0_Train.hdf5` | first 10% test, rest training pool |
| `burgers` | `train.hdf5` (10,000) and `test.hdf5` (2,000) | separate files |

The Burgers files were generated with PDEBench's Burgers solver (`Nu = 0.01`, which PDEBench
uses as ν = Nu/π) in the PDEBench layout: `tensor[N, T, X]`, `x-coordinate`, `t-coordinate`,
and the attribute `Nu`. To use PDEBench's own `1D_Burgers_Sols_Nu0.01.hdf5` instead, set
`train_file` to it, remove `test_file`, and add `test_ratio: 0.1`.

Pretraining reads only the first time step (or the coefficient field for Darcy); solution
trajectories of the training pool and all test samples are never used in Stage 1.

## Quick start

```bash
# supervised-only baseline
python run.py --config configs/advection.yaml --method supervised --seed 0 --data-root /path/to/1D/Advection/Train

# PI-Init: physics pretraining -> retain weights -> supervised fine-tuning
python run.py --config configs/advection.yaml --method pi_init --seed 0 --data-root /path/to/1D/Advection/Train
```

Useful options:

| Option | Meaning |
|---|---|
| `--budget B` | number of labeled trajectories, sampled uniformly at random from the training pool (default: whole pool) |
| `--n-cheap N` | number of cheap inputs used in pretraining (default: whole training pool) |
| `--extra-cheap file.npy` | append solver-free inputs, e.g. extra initial conditions from PDEBench's generators, at the reduced resolution |
| `--pi-epochs`, `--epochs` | override the pretraining / supervised epochs |
| `--device cpu` | run on CPU |

## Reproducing the paired comparison

```bash
# 3 seeds x {supervised, pi_init} for the selected datasets, then a summary table
DATASETS="advection" SEEDS="0 1 2" bash scripts/run_all.sh --data-root /path/to/1D/Advection/Train
python summarize.py --runs runs
```

`summarize.py` prints the mean ± SD test relative L2 error over seeds, the PI-only error
(the pretrained model before fine-tuning), and the relative improvement over supervised-only training.

Default settings (all four configs):

| | Stage 1 (physics) | Stage 2 (supervised) / baseline |
|---|---|---|
| Model | FNO, 4 Fourier layers, width 20, 12 modes | same |
| Optimizer | Adam, lr 1e-3, weight decay 1e-4, constant LR | fresh Adam, lr 1e-3, weight decay 1e-4, StepLR(100, 0.5) |
| Batch size / epochs | 256 / 50 | 256 / 500 |
| Data | cheap inputs only | labeled trajectories only |
| Rollout | autoregressive from u(t=0), `initial_step = 1` | same |

Grid reductions follow PDEBench (Burgers and Advection: space ×4, time ×5; 1D CFD: space ×8,
time ×5; Darcy: space ×2). Errors are the mean per-sample relative L2 error over all predicted
frames of the final-epoch model; no checkpoint is selected on the test set.

### Reference results

FNO test relative L2 error (mean ± SD over three seeds) reported in the paper with all labeled
training trajectories:

| Dataset | Supervised-only | PI-Init |
|---|---|---|
| Burgers | 0.0480 ± 0.0081 | 0.0224 ± 0.0031 |
| 1D Advection | 0.0660 ± 0.0490 | 0.0246 ± 0.0008 |
| 1D CFD | 0.1304 ± 0.0088 | 0.0818 ± 0.0009 |
| 2D Darcy | 0.1366 ± 0.0058 | 0.1233 ± 0.0065 |

The Burgers and Darcy PI-Init entries use N = 50,000 cheap inputs (training-pool inputs plus
solver-free samples, `--extra-cheap`); with the training pool alone the paper reports 0.0307
(Burgers) and 0.1299 (Darcy). Exact numbers depend on hardware and library versions.

Short runs are not indicative of the final comparison. On 1D CFD in particular, the
fine-tuning error of PI-Init first rises for a few epochs after pretraining and trails the
baseline early in training before overtaking it; use the full 500 supervised epochs.

## Outputs

Each run writes to `runs/<dataset>/<method>/B<budget>/seed<seed>/`:

| File | Content |
|---|---|
| `result.json` | final test relative L2, PI-only error, timing, labeled-set hash, full config |
| `history.csv` | supervised epochs: training loss and error, test error every `eval_every` epochs |
| `pi_history.csv`, `pi_checkpoint.pt` | pretraining loss per epoch and the retained weights (PI-Init only) |
| `model.pt` | final model |

## Code structure

```text
run.py                 command-line entry point for both methods
summarize.py           mean ± SD tables over seeds
configs/*.yaml         one config per dataset
pi_init/fno.py         FNO1d / FNO2d (from PDEBench, MIT)
pi_init/data.py        PDEBench HDF5 loading, test split, random labeled subsets, cheap inputs
pi_init/physics.py     autoregressive rollout, PDE residuals, per-dataset pretraining losses
pi_init/train.py       pretraining, supervised fine-tuning, evaluation
scripts/run_all.sh     paired runs over datasets and seeds
tests/                 residual checks and an end-to-end run on tiny synthetic files
```

To add a PDE, implement a `PhysicsLoss` subclass in `pi_init/physics.py` that maps a batch of
cheap inputs to a scalar residual loss, register it in `LOSSES`, and add a config.

## Tests

```bash
pip install pytest
pytest -q
```

The tests check the residuals against analytic solutions and run both methods end to end on
tiny synthetic HDF5 files on the CPU.

## Citation

```bibtex
@article{zhang2026cheapphysics,
  title   = {Rethinking How to Leverage Cheap Physics for Data-Efficient Neural Operators},
  author  = {Zhang, Shuo and Liang, Senwei},
  journal = {Neural Networks (under review)},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`). The FNO implementation is adapted from PDEBench and the original FNO code,
both MIT licensed (see `THIRD_PARTY_NOTICES.md`).

# CheapPhysics-NO

Code for **"Rethinking How to Leverage Cheap Physics for Data-Efficient Neural Operators"**.

**Physics-Informed Initialization (PI-Init)** pretrains a neural operator on cheap, unlabeled
PDE inputs with a physics-informed loss, retains the learned weights, and fine-tunes with the
supervised loss on a fixed random labeled subset. The supervised-only baseline uses the same
labels and training settings and differs only in its random initialization.

This repository provides the FNO implementation on four PDEBench problems:
Burgers, 1D Advection, 1D CFD, and 2D Darcy flow.

## Installation

```bash
git clone https://github.com/ShuoZHANG-BOT/CheapPhysics-NO.git
cd CheapPhysics-NO
pip install -r requirements.txt
```

## Data

Download the datasets from [PDEBench](https://github.com/pdebench/PDEBench) and set
`data.root` in `configs/<dataset>.yaml` (or pass `--data-root`).

## Usage

```bash
# supervised-only baseline
python run.py --config configs/burgers.yaml --method supervised --seed 0

# PI-Init: physics pretraining -> retain weights -> supervised fine-tuning
python run.py --config configs/burgers.yaml --method pi_init --seed 0
```

Run both methods over several seeds and summarize:

```bash
DATASETS="burgers advection cfd1d darcy" SEEDS="0 1 2" bash scripts/run_all.sh
```

Options such as `--budget` (number of labeled trajectories), `--n-cheap` (number of cheap
inputs), `--pi-epochs`, and `--epochs` override the config. Results are written to
`runs/<dataset>/<method>/B<budget>/seed<seed>/`.

## License

MIT. The FNO implementation is adapted from PDEBench (see `THIRD_PARTY_NOTICES.md`).

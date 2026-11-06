"""Collect result.json files and print mean +- SD per dataset, method, and label budget.

    python summarize.py --runs runs
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for path in sorted(args.runs.glob("*/*/B*/seed*/result.json")):
        result = json.loads(path.read_text())
        groups[(result["dataset"], result["budget"], result["method"])].append(result)
    if not groups:
        raise SystemExit(f"no results under {args.runs}")

    def stats(values):
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, sd

    print(f"{'dataset':<10} {'B':>6} {'method':<11} {'seeds':>5}  {'test rel. L2':>17}  {'PI-only':>8}  {'vs supervised':>13}")
    for (dataset, budget, method) in sorted(groups):
        rows = groups[(dataset, budget, method)]
        mean, sd = stats([r["test_relative_l2"] for r in rows])
        pi_only = [r["pi_only_test_relative_l2"] for r in rows if "pi_only_test_relative_l2" in r]
        pi_text = f"{statistics.mean(pi_only):.4f}" if pi_only else "-"
        base = groups.get((dataset, budget, "supervised"))
        gain = "-"
        if method != "supervised" and base:
            base_mean, _ = stats([r["test_relative_l2"] for r in base])
            gain = f"{100 * (base_mean - mean) / base_mean:+.1f}%"
        print(f"{dataset:<10} {budget:>6} {method:<11} {len(rows):>5}  {mean:.4f} +- {sd:.4f}  {pi_text:>8}  {gain:>13}")


if __name__ == "__main__":
    main()

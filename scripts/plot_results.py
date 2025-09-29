#!/usr/bin/env python3
"""
Plot aggregated learning curves from CSV logs in results_data.

- Expects CSV files named like: <env>__<simulation>__<seed>__<timestamp>.csv
  Example: starpilot__ppg_procgen__1__1758717352.csv
- Each CSV must have columns: "Wall time", "Step", "Value".

For each <simulation>, this script:
- Loads all seeds
- Applies EMA smoothing per seed
- Aligns on the union of Steps
- Computes mean and 95% CI across seeds
- Plots mean with shaded CI band

Usage:
  python scripts/plot_results.py \
    --data-dir /home/matteocederle/cleanrl/results_data \
    --output /home/matteocederle/cleanrl/results_data/plot.png \
    --ema-span 50
"""

import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Accept filenames like: <env>__<simulation>__<seed>__<timestamp>.csv
# - <env>: no underscores
# - <simulation>: may contain single underscores but not double underscores
# - <seed>: digits
# - <timestamp>: digits
FILENAME_RE = re.compile(
    r"^(?P<env>[^_]+)__(?P<sim>[^_]+(?:_[^_]+)*)__(?P<seed>\d+)__\d+\.csv$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aggregated results with EMA smoothing and 95% CI")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="results_data",
        help="Directory containing CSV result files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_data/plot.png",
        help="Path to save the generated plot",
    )
    parser.add_argument(
        "--ema-span",
        type=int,
        default=50,
        help="Span for exponential moving average smoothing (pandas ewm span)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Training Performance",
        help="Plot title",
    )
    return parser.parse_args()


def discover_files(data_dir: str) -> List[str]:
    files: List[str] = []
    for entry in os.listdir(data_dir):
        path = os.path.join(data_dir, entry)
        if not os.path.isfile(path):
            continue
        if entry.endswith(".csv") and FILENAME_RE.match(entry):
            files.append(path)
    files.sort()
    return files


def load_and_smooth(filepath: str, ema_span: int) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    # Validate required columns
    for col in ("Step", "Value"):
        if col not in df.columns:
            raise ValueError(f"File {filepath} is missing required column '{col}'")

    # Drop rows with NaNs in Step/Value
    df = df[["Step", "Value"]].dropna()

    # Sort by Step to ensure monotonic order before smoothing
    df = df.sort_values("Step")

    # Apply EMA smoothing to Value per run
    df["SmoothedValue"] = df["Value"].ewm(span=ema_span, adjust=False).mean()
    return df


def group_runs_by_sim(files: List[str]) -> Dict[str, List[Tuple[str, int]]]:
    groups: Dict[str, List[Tuple[str, int]]] = {}
    for path in files:
        name = os.path.basename(path)
        m = FILENAME_RE.match(name)
        if not m:
            continue
        sim = m.group("sim")
        seed = int(m.group("seed"))
        groups.setdefault(sim, []).append((path, seed))
    # Sort runs per sim by seed for consistency
    for sim in groups:
        groups[sim].sort(key=lambda t: t[1])
    return groups


def align_and_aggregate(runs: List[pd.DataFrame]) -> pd.DataFrame:
    """Align runs on union of Steps and compute mean and 95% CI.

    Returns a DataFrame with columns: Step, mean, lower, upper, count.
    """
    # Reindex each run on union of steps via outer join
    # Create a merged dataframe of smoothed values keyed by Step
    aligned = None
    value_columns: List[str] = []
    for idx, run_df in enumerate(runs):
        series = run_df.set_index("Step")["SmoothedValue"].copy()
        col_name = f"run_{idx+1}"
        value_columns.append(col_name)
        if aligned is None:
            aligned = series.to_frame(name=col_name)
        else:
            aligned = aligned.join(series.rename(col_name), how="outer")

    if aligned is None or aligned.empty:
        return pd.DataFrame(columns=["Step", "mean", "lower", "upper", "count"])  # type: ignore

    aligned = aligned.sort_index()

    # Compute stats across runs at each Step
    counts = aligned.count(axis=1)
    means = aligned.mean(axis=1, skipna=True)
    stds = aligned.std(axis=1, ddof=1, skipna=True)

    # Standard error and 95% CI (approx, using normal quantile 1.96)
    stderr = stds / np.sqrt(counts)
    ci = 1.96 * stderr
    lower = means - ci
    upper = means + ci

    out = pd.DataFrame(
        {
            "Step": aligned.index.values,
            "mean": means.values,
            "lower": lower.values,
            "upper": upper.values,
            "count": counts.values,
        }
    )
    return out


def plot_aggregates(aggregates: Dict[str, pd.DataFrame], title: str, dpi: int, output_path: str) -> None:
    plt.figure(figsize=(10, 6), dpi=dpi)

    # Color cycle for distinct simulations
    for sim_name, agg_df in sorted(aggregates.items()):
        if agg_df.empty:
            continue
        x = agg_df["Step"].to_numpy()
        y = agg_df["mean"].to_numpy()
        lo = agg_df["lower"].to_numpy()
        hi = agg_df["upper"].to_numpy()

        plt.plot(x, y, label=sim_name, linewidth=2)
        plt.fill_between(x, lo, hi, alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend(title="Simulation", frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    # Also display when run interactively
    try:
        plt.show()
    except Exception:
        pass


def main() -> None:
    args = parse_args()

    files = discover_files(args.data_dir)
    if not files:
        raise SystemExit(f"No CSV files found in {args.data_dir}")

    by_sim = group_runs_by_sim(files)
    if not by_sim:
        raise SystemExit("No files matched the expected naming pattern <env>__<sim>__<seed>__<timestamp>.csv")

    aggregates: Dict[str, pd.DataFrame] = {}
    for sim, items in by_sim.items():
        run_dfs: List[pd.DataFrame] = []
        for path, _seed in items:
            run_df = load_and_smooth(path, ema_span=args.ema_span)
            run_dfs.append(run_df)
        agg = align_and_aggregate(run_dfs)
        aggregates[sim] = agg

    plot_aggregates(aggregates, title=args.title, dpi=args.dpi, output_path=args.output)


if __name__ == "__main__":
    main()



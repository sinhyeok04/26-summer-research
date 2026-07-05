#!/usr/bin/env python3
"""
DR-Bearing Phase 1: Entropy visualization.
Usage:
    python scripts/plot_entropy.py <csv_path> [--out <png_path>] [--tau 0.5]

Plots step-index vs normalized entropy with:
  - threshold line at tau (default 0.5)
  - waypoint arrival markers
  - failure step highlighted (if max_steps reached)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "entropy_norm" not in df.columns:
        print(f"[ERROR] 'entropy_norm' column not found in {csv_path}.")
        print(f"  Available columns: {list(df.columns)}")
        print("  Did you run with --log_uncertainty?")
        sys.exit(1)
    return df


def plot_entropy(df: pd.DataFrame, tau: float, out_path: str, title: str) -> None:
    steps = df.index.to_numpy()
    entropy = df["entropy_norm"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    # ---- top: entropy curve ----
    ax = axes[0]
    ax.plot(steps, entropy, color="#2563eb", linewidth=1.5, label="entropy_norm")
    ax.axhline(tau, color="#dc2626", linewidth=1.0, linestyle="--", label=f"tau={tau}")
    ax.fill_between(steps, entropy, tau,
                    where=(entropy > tau), alpha=0.15, color="#dc2626",
                    label="high uncertainty zone")
    ax.set_ylabel("Normalized Entropy H [0,1]")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=10)

    # mark waypoint arrivals
    if "wp_index" in df.columns:
        wp_changes = df["wp_index"].diff().fillna(0) != 0
        for s in steps[wp_changes]:
            ax.axvline(s, color="#16a34a", linewidth=0.8, linestyle=":", alpha=0.7)

    # mark last step if it looks like failure (waypoint not reached at end)
    if "wp_index" in df.columns and "n_waypoints" in df.columns:
        last = df.iloc[-1]
        if last["wp_index"] < last["n_waypoints"] - 1:
            ax.axvline(steps[-1], color="#f97316", linewidth=1.5, linestyle="-",
                       label="failure (max_steps)")
            ax.legend(loc="upper right", fontsize=8)

    # ---- bottom: u (uncertainty score) ----
    ax2 = axes[1]
    if "u" in df.columns:
        ax2.bar(steps, df["u"].to_numpy(), color="#7c3aed", alpha=0.6, width=1.0)
        ax2.axhline(0.5, color="#dc2626", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("u (uncertainty)")
        ax2.set_ylim(-0.05, 1.05)
    else:
        ax2.text(0.5, 0.5, "u column not found", transform=ax2.transAxes, ha="center")

    ax2.set_xlabel("Step index")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DR-Bearing entropy from CSV")
    parser.add_argument("csv_path", help="Path to *_uav_traj_records.csv")
    parser.add_argument("--out", default=None, help="Output PNG path (default: same dir as CSV)")
    parser.add_argument("--tau", type=float, default=0.5, help="Entropy threshold line (default 0.5)")
    args = parser.parse_args()

    csv_path = Path(args.csv_path).resolve()
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    out_path = args.out or str(csv_path.parent / (csv_path.stem + "_entropy.png"))
    df = load_csv(str(csv_path))

    stem = csv_path.stem
    plot_entropy(df, tau=args.tau, out_path=out_path, title=stem)


if __name__ == "__main__":
    main()

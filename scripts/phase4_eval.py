#!/usr/bin/env python3
"""
Phase 4: Full 8-route evaluation across 5 modes.

Modes
-----
  off      : baseline model-only (reuses existing baseline runs — no re-execution)
  v2       : Innovation-Gated KF (gate only)
  v2_hl    : KF gate + heading-rate limiter
  v2_ema   : KF gate + VO scale EMA
  v2_full  : KF gate + heading limiter + scale EMA  (full v2 stack)

Routes: 34bc_{50,51}  36bc_{50,51}  37bc_{50,51}  38bc_{50,51}

Usage
-----
  # Run all new modes, generate report
  conda run -n bearing_env python scripts/phase4_eval.py

  # Run only specific modes/routes
  conda run -n bearing_env python scripts/phase4_eval.py --modes v2 v2_hl --routes 34bc_50 38bc_51

  # Force re-run (ignore manifest)
  conda run -n bearing_env python scripts/phase4_eval.py --force

Output
------
  loc2traj/phase4_manifest.json   — mode+route → output dir mapping
  docs/phase4_results.md          — NE comparison table
"""

import os
import sys
import json
import math
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJ_DIR = Path(__file__).resolve().parent.parent
LOC2TRAJ_DIR = PROJ_DIR / "loc2traj"
MANIFEST_PATH = LOC2TRAJ_DIR / "phase4_manifest.json"

ROUTES = [
    ("34bc", 50), ("34bc", 51),
    ("36bc", 50), ("36bc", 51),
    ("37bc", 50), ("37bc", 51),
    ("38bc", 50), ("38bc", 51),
]
ROUTE_KEYS = [f"{r}_{t}" for r, t in ROUTES]

FAIL_ROUTES = {"34bc_50", "36bc_50", "38bc_50", "38bc_51"}

# Baseline dirs from Phase 1 (mode='off', 2D). Reused for 2D runs only.
BASELINE_DIRS_2D = {
    "34bc_50": "nav_34bc50_s25_t20_d2d_phr5_20260702_215635",
    "34bc_51": "nav_34bc51_s25_t20_d2d_phr5_20260703_144143",
    "36bc_50": "nav_36bc50_s25_t20_d2d_phr5_20260703_150933",
    "36bc_51": "nav_36bc51_s25_t20_d2d_phr5_20260703_155859",
    "37bc_50": "nav_37bc50_s25_t20_d2d_phr5_20260703_161233",
    "37bc_51": "nav_37bc51_s25_t20_d2d_phr5_20260703_161555",
    "38bc_50": "nav_38bc50_s25_t20_d2d_phr5_20260703_161936",
    "38bc_51": "nav_38bc51_s25_t20_d2d_phr5_20260703_163126",
}

# Extra CLI flags per mode (appended to the common base command)
MODE_FLAGS = {
    "off":     [],
    "v2":      ["--fusion_mode", "v2"],
    "v2_ar":   ["--fusion_mode", "v2", "--use_adaptive_r"],
    "v2_hl":   ["--fusion_mode", "v2", "--use_heading_limit"],
    "v2_ema":  ["--fusion_mode", "v2", "--use_scale_ema"],
    "v2_full": ["--fusion_mode", "v2", "--use_heading_limit", "--use_scale_ema"],
}
ALL_MODES = list(MODE_FLAGS.keys())

# KF hyper-parameters (calibrated in Phase 1)
KF_FLAGS = [
    "--kf_q", "100.0",
    "--kf_r", "600.0",
    "--chi2_gate", "5.99",
    "--reanchor_after", "5",
    "--consensus_radius", "15.0",
    "--vo_scale_init", "1.2",
    "--ema_alpha", "0.1",
    "--hunting_radius", "50.0",
    "--max_heading_rate", "30.0",
]

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def haversine_m(lon1, lat1, lon2, lat2):
    R = 6_371_000.0
    la1, la2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    dlat = la2 - la1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0.0, a)))

# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def manifest_key(mode, route_key, dim="2d"):
    suffix = f"_{dim}" if dim != "2d" else ""
    return f"{mode}__{route_key}{suffix}"


def report_path(dim="2d"):
    return PROJ_DIR / "docs" / f"phase4_results{'_' + dim if dim != '2d' else ''}.md"

# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def find_csv(out_dir: Path, route_key: str) -> Path:
    candidates = list(out_dir.glob("*records.csv"))
    if not candidates:
        raise FileNotFoundError(f"No records CSV in {out_dir}")
    # Prefer exact route_key match
    for c in candidates:
        if route_key.replace("bc_", "bc_") in c.name:
            return c
    return candidates[0]


def compute_metrics(out_dir: Path, route_key: str) -> dict:
    csv_path = find_csv(out_dir, route_key)
    df = pd.read_csv(csv_path)

    ne_vals = [
        haversine_m(row.cur_lon_pred, row.cur_lat_pred,
                    row.cur_lon_real, row.cur_lat_real)
        for _, row in df.iterrows()
    ]
    ne_mean = float(sum(ne_vals) / len(ne_vals)) if ne_vals else float("nan")

    # Waypoint completion: how many WPs reached vs total in the trajectory
    max_wp_reached = int(df["waypoint_index"].max())
    any_reached = df["reached_waypoint"].any()
    # Count distinct WPs actually ticked (reached_waypoint==True)
    wps_reached = int(df[df["reached_waypoint"] == True]["waypoint_index"].nunique())

    steps = len(df)

    metrics = {
        "ne_m": round(ne_mean, 1),
        "steps": steps,
        "max_wp": max_wp_reached,
        "wps_reached": wps_reached,
        "csv": str(csv_path),
    }

    # KF diagnostics (only for v2 modes)
    if "kf_d2" in df.columns:
        d2 = df["kf_d2"].dropna()
        acc = df["kf_accept"].dropna()
        metrics["reject_pct"] = round(100.0 * (1 - acc.mean()), 1) if len(acc) else float("nan")
        metrics["d2_mean"] = round(float(d2.mean()), 2) if len(d2) else float("nan")
    else:
        metrics["reject_pct"] = None
        metrics["d2_mean"] = None

    return metrics

# ---------------------------------------------------------------------------
# Nav runner
# ---------------------------------------------------------------------------

def run_nav(rsi_id: str, traj_id: int, extra_flags: list, dim: str = "2d") -> Path:
    """
    Run nav.py and return the output directory created during this run.
    We snapshot loc2traj before/after to identify the new directory.
    """
    before = set(LOC2TRAJ_DIR.iterdir())

    if dim == "3d":
        model_flags = [
            "--uav_2d3d", "3d",
            "--cvphr_3d_best_model_dir", str(PROJ_DIR / "Bearing_UAV" / "cross_view"),
        ]
    else:
        model_flags = [
            "--uav_2d3d", "2d",
            "--cvphr_2d_best_model_dir", str(PROJ_DIR / "Bearing_UAV" / "satellite_view"),
        ]

    base_cmd = [
        "conda", "run", "-n", "bearing_env",
        "python", "-m", "naver.runners.nav",
        "--rsi_id", rsi_id,
        "--traj_id", str(traj_id),
        "--uav_step", "25",
        "--th_arrive", "20",
    ] + model_flags + KF_FLAGS + extra_flags

    print(f"  $ {' '.join(base_cmd[-10:])}")  # print tail of command for readability
    result = subprocess.run(
        base_cmd,
        cwd=str(PROJ_DIR),
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nav.py exited with code {result.returncode}")

    after = set(LOC2TRAJ_DIR.iterdir())
    new_dirs = [p for p in (after - before) if p.is_dir()]
    if not new_dirs:
        raise RuntimeError("nav.py completed but no new directory found in loc2traj/")
    if len(new_dirs) > 1:
        # pick most recently modified
        new_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return new_dirs[0]

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_phase4(modes, route_keys, force=False, dim="2d"):
    manifest = load_manifest()

    for mode in modes:
        for route_key in route_keys:
            mkey = manifest_key(mode, route_key, dim)
            rsi_id, traj_id = route_key.rsplit("_", 1)
            traj_id = int(traj_id)

            # --- off mode with 2D: reuse pre-existing baseline dirs ---
            if mode == "off" and dim == "2d":
                baseline_dir = LOC2TRAJ_DIR / BASELINE_DIRS_2D[route_key]
                if not baseline_dir.exists():
                    print(f"[WARN] Baseline dir missing for {route_key}: {baseline_dir}")
                    continue
                manifest[mkey] = str(baseline_dir)
                save_manifest(manifest)
                print(f"[skip] {mode}/{route_key} [{dim}]  (baseline reused)")
                continue

            # --- all other modes (including off+3d) ---
            if mkey in manifest and not force:
                out_dir = Path(manifest[mkey])
                if out_dir.exists():
                    print(f"[skip] {mode}/{route_key} [{dim}]  (manifest: {out_dir.name})")
                    continue
                else:
                    print(f"[warn] {mode}/{route_key} [{dim}]  manifest entry invalid, re-running")

            print(f"\n[run] {mode}/{route_key} [{dim}]  {datetime.now().strftime('%H:%M:%S')}")
            try:
                out_dir = run_nav(rsi_id, traj_id, MODE_FLAGS[mode], dim=dim)
                manifest[mkey] = str(out_dir)
                save_manifest(manifest)
                print(f"  → {out_dir.name}")
            except Exception as e:
                print(f"  [ERROR] {e}")

    return manifest

# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(manifest, dim="2d"):
    print("\n=== Computing metrics ===")
    results = {}  # (mode, route_key) → metrics dict

    for mode in ALL_MODES:
        for route_key in ROUTE_KEYS:
            mkey = manifest_key(mode, route_key, dim)
            if mkey not in manifest:
                continue
            out_dir = Path(manifest[mkey])
            if not out_dir.exists():
                print(f"  [WARN] {mode}/{route_key}: output dir missing")
                continue
            try:
                m = compute_metrics(out_dir, route_key)
                results[(mode, route_key)] = m
                tag = "FAIL" if route_key in FAIL_ROUTES else "OK  "
                rej = f"{m['reject_pct']}%" if m['reject_pct'] is not None else "—"
                print(f"  {mode:10s} {route_key} [{tag}]  NE={m['ne_m']:6.1f}m  WP={m['wps_reached']}  rej={rej}")
            except Exception as e:
                print(f"  [ERROR] {mode}/{route_key}: {e}")

    model_label = "satellite_view (2D/nadir)" if dim == "2d" else "cross_view (3D/oblique)"

    # Build table
    lines = [
        "# Phase 4: 8-Route Evaluation Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {model_label}",
        "",
        "KF params: Q=100 m², R=600 m², gate=5.99, hunting_radius=50m, max_heading_rate=30°, ema_alpha=0.1",
        "",
        "## NE per Route (metres)",
        "",
    ]

    # Header
    header = "| route | status |" + "".join(f" {m} |" for m in ALL_MODES)
    sep    = "| --- | --- |" + "".join(" --- |" for _ in ALL_MODES)
    lines += [header, sep]

    for route_key in ROUTE_KEYS:
        status = "FAIL" if route_key in FAIL_ROUTES else "OK"
        cells = []
        for mode in ALL_MODES:
            m = results.get((mode, route_key))
            cells.append(f"{m['ne_m']:.1f}" if m else "—")
        lines.append(f"| {route_key} | {status} |" + "".join(f" {c} |" for c in cells))

    lines += [""]

    # Averages
    lines += ["## Averages by Group", ""]
    avg_header = "| group |" + "".join(f" {m} |" for m in ALL_MODES)
    avg_sep    = "| --- |" + "".join(" --- |" for _ in ALL_MODES)
    lines += [avg_header, avg_sep]

    for label, keys in [("FAIL routes", FAIL_ROUTES), ("OK routes", set(ROUTE_KEYS) - FAIL_ROUTES), ("All routes", set(ROUTE_KEYS))]:
        cells = []
        for mode in ALL_MODES:
            vals = [results[(mode, k)]["ne_m"] for k in keys if (mode, k) in results]
            cells.append(f"{sum(vals)/len(vals):.1f}" if vals else "—")
        lines.append(f"| {label} |" + "".join(f" {c} |" for c in cells))

    lines += [""]

    # Gate rejection summary for v2 modes
    v2_modes = [m for m in ALL_MODES if m != "off"]
    lines += ["## Gate Rejection Rate (v2 modes only)", ""]
    rej_header = "| route |" + "".join(f" {m} |" for m in v2_modes)
    rej_sep    = "| --- |" + "".join(" --- |" for _ in v2_modes)
    lines += [rej_header, rej_sep]

    for route_key in ROUTE_KEYS:
        cells = []
        for mode in v2_modes:
            m = results.get((mode, route_key))
            cells.append(f"{m['reject_pct']}%" if m and m["reject_pct"] is not None else "—")
        lines.append(f"| {route_key} |" + "".join(f" {c} |" for c in cells))

    rpath = report_path(dim)
    report = "\n".join(lines) + "\n"
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text(report)
    print(f"\nReport saved → {rpath}")
    print("\n" + report)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 4: 8-route × 5-mode evaluation")
    p.add_argument("--modes", nargs="+", default=ALL_MODES,
                   choices=ALL_MODES, help="Which modes to run (default: all)")
    p.add_argument("--routes", nargs="+", default=ROUTE_KEYS,
                   choices=ROUTE_KEYS, help="Which routes to evaluate (default: all)")
    p.add_argument("--uav_dim", default="2d", choices=["2d", "3d"],
                   help="Model dimension: 2d=satellite_view, 3d=cross_view (default: 2d)")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if manifest entry exists")
    p.add_argument("--report_only", action="store_true",
                   help="Skip nav runs, only recompute report from existing manifest")
    return p.parse_args()


def main():
    args = parse_args()

    LOC2TRAJ_DIR.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        manifest = run_phase4(args.modes, args.routes, force=args.force, dim=args.uav_dim)
    else:
        manifest = load_manifest()

    generate_report(manifest, dim=args.uav_dim)


if __name__ == "__main__":
    main()

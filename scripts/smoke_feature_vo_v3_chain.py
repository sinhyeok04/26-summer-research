"""
Feature VO Phase 0 -- v3: chain ORB/homography's own rotation estimate.

v2 validated displacement MAGNITUDE only (theta-free) and left direction as an
open gap: composing an ego-frame displacement into world (lon, lat) needs SOME
absolute heading, and the logged `theta` was shown unreliable for 3d frames.

v3 tests the natural next hypothesis: trust ORB/homography's own *relative*
rotation estimate (est_yaw_deg) between consecutive frames, and chain it --

    heading(i) = heading(i-1) + est_yaw_deg(pair i-1 -> i)

starting from ONE trustworthy seed per route: the true initial travel
direction, computed directly from real (lon, lat) at frames 0 and 1 (this is
simulator ground truth, not derived from any image, so it isn't circular the
way using `theta` would be -- it stands in for "the one absolute reference a
real deployment would get from compass/IMU/first fix").

From there, each step's world-frame vector is composed using the CHAIN's
heading (not the logged theta), and compared against the true (east, north)
step vector -- now a full magnitude+direction check, not magnitude-only. Step
1 of every route is excluded from the aggregate error (it's seeded from
truth, so it's accurate by construction, not a measurement of the method).

Usage:
    conda activate bearing_env
    python scripts/smoke_feature_vo_v3_chain.py --uav_2d3d 3d
"""
import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config.base_info import PATCH_SIZE
from naver.fusion.innovation_gated import _deg_to_m
from scripts.smoke_latent_vo import collect_pairs, wrap180, ego_to_world_m
from scripts.homography_vo import build_intrinsics
from scripts.smoke_feature_vo_v2 import eval_pair as eval_pair_v2


ROT_SIGN = -1.0  # both signs were tested empirically; -1 chains to lower error, though neither is good (see report)


def true_vector(prev: Dict, cur: Dict):
    ref_lat = (prev["lat"] + cur["lat"]) / 2.0
    e, n = _deg_to_m(cur["lon"] - prev["lon"], cur["lat"] - prev["lat"], ref_lat)
    return e, n


def group_routes(pairs: List[Dict]) -> Dict[str, List[Dict]]:
    routes: Dict[str, List[Dict]] = {}
    for p in pairs:
        routes.setdefault(p["route"], []).append(p)
    for k in routes:
        routes[k].sort(key=lambda p: p["prev"]["frame"])
    return routes


def run_chain(route_pairs: List[Dict], K, args) -> List[Dict]:
    """route_pairs: consecutive (frame_i -> frame_i+1) pairs for one route, in order."""
    if not route_pairs:
        return []

    # Seed: true travel direction of the FIRST step in this chain (ground truth,
    # not image-derived -- stands in for a real deployment's initial fix/compass).
    p0 = route_pairs[0]
    seed_e, seed_n = true_vector(p0["prev"], p0["cur"])
    heading = math.degrees(math.atan2(seed_n, seed_e))  # CCS convention: 0=east, CCW+

    rows = []
    for i, pair in enumerate(route_pairs):
        prev, cur = pair["prev"], pair["cur"]
        m_per_px = prev["viewspan_m"] / PATCH_SIZE
        true_e, true_n = true_vector(prev, cur)
        true_dist = math.hypot(true_e, true_n)

        r = eval_pair_v2(pair, K, args)  # reuses v2's ORB+homography+reprojection-gate pipeline

        row = {"route": pair["route"], "frame": cur["frame"], "step_idx": i,
               "true_dist_m": true_dist, "heading_before_deg": heading, "valid": r["valid"]}

        if i == 0:
            # seeded step: composed vector == true vector by construction, not a real measurement
            row["seeded"] = True
            row["err_m"] = 0.0
            heading_after = heading  # no rotation info to chain from a seeded step's own yaw is still useful below
            if r["valid"] and r.get("est_yaw_deg") is not None:
                heading_after = heading + ROT_SIGN * r["est_yaw_deg"]
            heading = heading_after
            rows.append(row)
            continue

        row["seeded"] = False
        if not r["valid"] or r.get("est_yaw_deg") is None:
            row["err_m"] = None
            row["reason"] = r["reason"]
            # can't advance the chain's heading without an estimate; carry forward unchanged
            rows.append(row)
            continue

        # advance heading by this step's estimated relative rotation, then compose displacement with it
        heading_after = heading + ROT_SIGN * r["est_yaw_deg"]
        if r.get("dx_px") is None:
            row["err_m"] = None
            row["reason"] = "no_displacement"
            heading = heading_after
            rows.append(row)
            continue

        # compose the ego-frame pixel displacement into world (east, north) using the CHAIN's
        # heading (heading_after = this step's destination-frame absolute heading estimate),
        # instead of the unreliable logged theta.
        east_m, north_m = ego_to_world_m(r["dx_px"], r["dy_px"], m_per_px, heading_after)
        pred_vec = np.array([east_m, north_m])
        true_vec = np.array([true_e, true_n])
        row["err_m"] = float(np.linalg.norm(pred_vec - true_vec))
        row["heading_after_deg"] = heading_after
        heading = heading_after
        rows.append(row)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uav_2d3d", default="3d", choices=["2d", "3d"])
    ap.add_argument("--num_pairs", type=int, default=40)
    ap.add_argument("--n_features", type=int, default=3000)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac_thresh", type=float, default=5.0)
    ap.add_argument("--min_matches", type=int, default=30)
    ap.add_argument("--min_inlier_ratio", type=float, default=0.5)
    ap.add_argument("--min_valid_frac", type=float, default=0.3)
    ap.add_argument("--max_warp_mae", type=float, default=20.0)
    ap.add_argument("--altitude_m", type=float, default=204.5257)
    args = ap.parse_args()

    proj_root = Path(__file__).resolve().parents[1]
    loc2traj_dir = proj_root / "loc2traj"
    pairs = collect_pairs(loc2traj_dir, args.uav_2d3d, args.num_pairs)
    routes = group_routes(pairs)
    print(f"{len(pairs)} pairs across {len(routes)} routes.")

    K = build_intrinsics(PATCH_SIZE, PATCH_SIZE, args.altitude_m, 64.0 / PATCH_SIZE)

    all_rows = []
    for route, route_pairs in routes.items():
        rows = run_chain(route_pairs, K, args)
        all_rows.extend(rows)
        for row in rows:
            err = row.get("err_m")
            err_s = f"{err:.1f}m" if err is not None else "-"
            tag = "seed" if row.get("seeded") else row.get("reason", "ok")
            print(f"  {route} step {row['step_idx']} (frame {row['frame']}): true={row['true_dist_m']:.1f}m err={err_s} [{tag}]")

    measured = [r["err_m"] for r in all_rows if not r.get("seeded") and r.get("err_m") is not None]
    n_valid_chain_steps = len(measured)
    n_total_chain_steps = sum(1 for r in all_rows if not r.get("seeded"))
    print(f"\nChained (non-seeded) steps with a usable estimate: {n_valid_chain_steps}/{n_total_chain_steps}")
    if measured:
        arr = np.array(measured)
        print(f"Full vector error (magnitude + chained-heading direction): mean={arr.mean():.2f}m "
              f"std={arr.std():.2f}m median={np.median(arr):.2f}m")
    else:
        print("No usable chained steps.")


if __name__ == "__main__":
    main()

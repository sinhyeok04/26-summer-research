"""
Feature VO Phase 0 smoke test, v2 -- theta-free validation.

Why a v2: docs/feature_vo_smoke_report.md (v1) judged ORB+RANSAC by composing
its ego-frame estimate with the nav log's `theta` into world lon/lat, then
compared against world lon/lat computed the same way. That FAILed (error std
13.17 m). But a follow-up check in this session found the opposite: warping
frame t-1 by the estimated homography and directly overlaying it on frame t
gives near-seamless pixel alignment (mean abs diff ~7-17 out of 255, vs ~45-70
for the unwarped pair) across most pairs -- direct evidence the ORB+RANSAC
correspondences ARE finding the real geometric relationship between the two
images. The two findings are reconciled by: 3d mode's target patches come
from `_find_nearest_v3d` (nav.py), a nearest-(lat,lng)-only library lookup
with no heading field at all -- so the logged `theta` likely does not
describe the actual orientation the v3d image was captured/rendered at. v1's
"ground truth" was the unreliable part, not the estimator.

v2 therefore:
  - trusts ORB+RANSAC's own rotation estimate instead of the logged theta,
  - validates against the one thing that IS still reliable regardless of
    which v3d image got picked -- the true displacement MAGNITUDE, derived
    from consecutive real (lon, lat) in the flight log (that reflects the
    simulated flight physics, not image selection),
  - replaces "compare estimated world-frame vector to logged theta" with a
    self-consistency gate: does warping frame t-1 by the estimated H actually
    reproduce frame t? (reprojection MAE)

Direction/world-frame accuracy is explicitly NOT validated here -- there is
currently no trustworthy ground truth for it in 3d mode. See "Known gap" in
the generated report.

Usage:
    conda activate bearing_env
    python scripts/smoke_feature_vo_v2.py --uav_2d3d 3d --num_pairs 40
"""
import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from config.base_info import PATCH_SIZE
from naver.fusion.innovation_gated import _deg_to_m
from scripts.smoke_latent_vo import collect_pairs, wrap180
from scripts.homography_vo import load_gray, match_features, estimate_homography, decompose_and_select, \
    build_intrinsics, rotation_matrix_to_euler_deg


# ---------------------------------------------------------------------------
# The two theta-free measurements
# ---------------------------------------------------------------------------

def warp_reprojection_mae(gray1: np.ndarray, gray2: np.ndarray, H: np.ndarray):
    """Warp frame t-1 into frame t's pixel frame with the estimated H and compare directly to frame t.
    This is the self-consistency check that doesn't need theta at all."""
    h, w = gray2.shape[:2]
    warped1 = cv2.warpPerspective(gray1, H, (w, h))
    valid = warped1 > 0
    valid_frac = float(valid.mean())
    if valid.sum() == 0:
        return float("inf"), 0.0
    diff = np.abs(gray2.astype(np.float32) - warped1.astype(np.float32))
    mae = float(diff[valid].mean())
    return mae, valid_frac


def displacement_via_homography(H: np.ndarray, w: int, h: int):
    """Map the patch CENTER through the raw pixel homography H (projective, so divide
    by the homogeneous w-component) -- gives how far the view shifted, in pixels, with
    no camera-intrinsics / plane-depth assumption needed (unlike decomposeHomographyMat's t)."""
    cx, cy = w / 2.0, h / 2.0
    p = H @ np.array([cx, cy, 1.0])
    p = p[:2] / p[2]
    return float(p[0] - cx), float(p[1] - cy)


# ---------------------------------------------------------------------------
# Per-pair evaluation
# ---------------------------------------------------------------------------

def eval_pair(pair: Dict, K: np.ndarray, args) -> Dict:
    prev, cur = pair["prev"], pair["cur"]
    img1, gray1 = load_gray(prev["path"])
    img2, gray2 = load_gray(cur["path"])
    h, w = gray1.shape[:2]
    m_per_px = prev["viewspan_m"] / PATCH_SIZE

    ref_lat = (prev["lat"] + cur["lat"]) / 2.0
    true_e, true_n = _deg_to_m(cur["lon"] - prev["lon"], cur["lat"] - prev["lat"], ref_lat)
    true_dist = math.hypot(true_e, true_n)
    dtheta_logged = wrap180(cur["theta"] - prev["theta"])  # logged only, NOT used for validation

    result = {
        "run": pair["run"], "frame": cur["frame"], "true_dist_m": true_dist,
        "dtheta_logged_deg": dtheta_logged, "valid": False, "reason": "unknown",
    }

    kp1, kp2, good = match_features(gray1, gray2, n_features=args.n_features, ratio=args.ratio)
    result["n_good"] = len(good)
    if len(good) < args.min_matches:
        result["reason"] = "few_matches"
        return result

    H, mask, pts1, pts2 = estimate_homography(kp1, kp2, good, ransac_thresh=args.ransac_thresh)
    if H is None:
        result["reason"] = "homography_failed"
        return result

    inlier_ratio = float(mask.sum()) / len(good)
    result["inlier_ratio"] = inlier_ratio
    if inlier_ratio < args.min_inlier_ratio:
        result["reason"] = "low_inlier_ratio"
        return result

    mae, valid_frac = warp_reprojection_mae(gray1, gray2, H)
    result["warp_mae"] = mae
    result["warp_valid_frac"] = valid_frac
    if valid_frac < args.min_valid_frac or mae > args.max_warp_mae:
        result["reason"] = "poor_reprojection"
        return result

    dx_px, dy_px = displacement_via_homography(H, w, h)
    pred_dist = math.hypot(dx_px, dy_px) * m_per_px
    result["pred_dist_m"] = pred_dist
    result["mag_err_m"] = pred_dist - true_dist  # signed: + means overestimate
    result["dx_px"] = dx_px
    result["dy_px"] = dy_px
    result["H"] = H

    # Rotation: logged for reference only -- no trustworthy ground truth to check it against.
    try:
        inlier_mask = mask.ravel().astype(bool)
        R, t, n, chosen_idx, n_solutions = decompose_and_select(H, K, pts1, pts2, mask)
        roll, pitch, yaw = rotation_matrix_to_euler_deg(R)
        result["est_yaw_deg"] = yaw
    except Exception:
        result["est_yaw_deg"] = None

    result["valid"] = True
    result["reason"] = "ok"
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def stat(values: List[Optional[float]]) -> Dict[str, float]:
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "median": float(np.median(arr)), "n": len(arr)}


def write_report(out_path: Path, uav_2d3d: str, results: List[Dict], reason_counts: Dict[str, int],
                  n_total: int, verdict: str, verdict_reason: str, args):
    lines = []
    lines.append("# Feature VO Phase 0 Smoke Report -- v2 (theta-free)\n")
    lines.append(f"- Mode: `{uav_2d3d}` · pairs: {n_total} (same harness/pairs as v1 and the latent report)")
    lines.append("- v1 (`docs/feature_vo_smoke_report.md`) judged ORB+RANSAC against a world-frame ground truth "
                 "composed using the nav log's `theta`. This session found that `theta` is not trustworthy for "
                 "3d mode's `_find_nearest_v3d`-sourced frames (no heading field in the v3d library at all), and "
                 "that warping frame t-1 by the estimated homography reproduces frame t almost exactly on most "
                 "pairs -- direct evidence the estimator itself is largely correct. v2 drops `theta` from "
                 "validation entirely.\n")

    lines.append("## What's validated here, and what isn't\n")
    lines.append("| | v1 | v2 |")
    lines.append("|---|---|---|")
    lines.append("| Validity gate | n_matches, inlier_ratio | + reprojection MAE (does warp(frame t-1, H) actually match frame t?) |")
    lines.append("| Accuracy check | world-frame (east, north) vs. theta-composed ground truth | displacement **magnitude** vs. true step distance (from real lon/lat, theta-independent) |")
    lines.append("| Direction / heading | compared to logged theta (rejected as unreliable) | logged for reference only, not validated |")
    lines.append("")

    plain_valid = [r for r in results if r["valid"]]
    valid_rate = len(plain_valid) / n_total if n_total else 0.0
    mag_errs = [r["mag_err_m"] for r in plain_valid]
    mag_stat_signed = stat(mag_errs)
    mag_stat_abs = stat([abs(e) for e in mag_errs])
    warp_mae_stat = stat([r.get("warp_mae") for r in plain_valid])

    lines.append("## Valid rate and rejection reasons\n")
    lines.append(f"- valid rate: **{100*valid_rate:.0f}%** ({len(plain_valid)}/{n_total})")
    lines.append("| reason | count |")
    lines.append("|---|---|")
    for k, v in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Displacement magnitude error (valid pairs, metres, theta-independent)\n")
    lines.append("| | mean (signed bias) | std | mean abs | median abs | n |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| magnitude error | {mag_stat_signed['mean']:+.2f} | {mag_stat_signed['std']:.2f} | "
                 f"{mag_stat_abs['mean']:.2f} | {mag_stat_abs['median']:.2f} | {mag_stat_signed['n']} |")
    lines.append("")
    lines.append(f"Reprojection MAE on valid pairs: mean {warp_mae_stat['mean']:.2f}, "
                 f"median {warp_mae_stat['median']:.2f} (0-255 scale; for reference, raw unwarped-pair MAE runs "
                 "40-70 in this dataset, so anything well below that indicates real alignment, not chance).\n")

    lines.append("## Verdict\n")
    lines.append(f"**{verdict}**\n")
    lines.append(verdict_reason + "\n")

    lines.append("## Known gap -- direction is still unvalidated\n")
    lines.append("This report does not establish that the estimated **direction** (heading-composed world-frame "
                 "vector) is correct -- only that the magnitude is, and that the homography self-consistently "
                 "reprojects. Composing a displacement into world (lon, lat) still requires *some* heading "
                 "estimate, and the only two candidates are (a) the logged `theta` (shown unreliable) or (b) "
                 "ORB/homography's own rotation estimate, e.g. `est_yaw_deg` logged per-pair below (untested "
                 "against any ground truth). Before this can feed the fusion filter, direction needs an "
                 "independent validation path -- e.g. chaining several steps and checking the resulting "
                 "trajectory shape against the known waypoint path, or finding/recovering true per-image "
                 "capture heading if it exists anywhere in the v3d generation pipeline.\n")

    lines.append("## Per-pair detail\n")
    lines.append("| run | frame | true (m) | pred (m) | mag err (m) | warp MAE | inlier% | n_good | est_yaw | logged Δθ | reason |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        run_short = r["run"].split("_s25")[0].replace("nav_", "")
        pred = f"{r['pred_dist_m']:.1f}" if r.get("pred_dist_m") is not None else "-"
        me = f"{r['mag_err_m']:+.1f}" if r.get("mag_err_m") is not None else "-"
        wm = f"{r['warp_mae']:.1f}" if r.get("warp_mae") is not None else "-"
        inl = f"{100*r['inlier_ratio']:.0f}" if r.get("inlier_ratio") is not None else "-"
        yaw = f"{r['est_yaw_deg']:+.1f}" if r.get("est_yaw_deg") is not None else "-"
        lines.append(f"| {run_short} | {r['frame']} | {r['true_dist_m']:.1f} | {pred} | {me} | {wm} | {inl} | "
                     f"{r['n_good']} | {yaw} | {r['dtheta_logged_deg']:+.1f} | {r['reason']} |")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uav_2d3d", default="3d", choices=["2d", "3d"])
    ap.add_argument("--num_pairs", type=int, default=40)
    ap.add_argument("--n_features", type=int, default=3000)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac_thresh", type=float, default=5.0)
    ap.add_argument("--min_matches", type=int, default=30)
    ap.add_argument("--min_inlier_ratio", type=float, default=0.5)
    ap.add_argument("--min_valid_frac", type=float, default=0.3, help="min fraction of frame t-1 that lands inside frame t after warping")
    ap.add_argument("--max_warp_mae", type=float, default=20.0, help="reprojection MAE gate (0-255 scale)")
    ap.add_argument("--altitude_m", type=float, default=204.5257)
    ap.add_argument("--out", default="docs/feature_vo_smoke_report_v2.md")
    args = ap.parse_args()

    proj_root = Path(__file__).resolve().parents[1]
    loc2traj_dir = proj_root / "loc2traj"
    pairs = collect_pairs(loc2traj_dir, args.uav_2d3d, args.num_pairs)
    print(f"Collected {len(pairs)} pairs (mode={args.uav_2d3d}).")

    K = build_intrinsics(PATCH_SIZE, PATCH_SIZE, args.altitude_m, 64.0 / PATCH_SIZE)

    results = []
    for i, pair in enumerate(pairs):
        r = eval_pair(pair, K, args)
        results.append(r)
        pred = f"{r['pred_dist_m']:.1f}" if r.get("pred_dist_m") is not None else "-"
        wm = f"{r['warp_mae']:.1f}" if r.get("warp_mae") is not None else "-"
        print(f"[{i+1}/{len(pairs)}] {pair['run']} frame {pair['cur']['frame']}: true={r['true_dist_m']:.1f}m "
              f"pred={pred}m warp_mae={wm} ({r['reason']})")

    reason_counts: Dict[str, int] = {}
    for r in results:
        reason_counts[r["reason"]] = reason_counts.get(r["reason"], 0) + 1

    n_total = len(results)
    valid = [r for r in results if r["valid"]]
    valid_rate = len(valid) / n_total if n_total else 0.0
    mag_errs = [r["mag_err_m"] for r in valid]
    err_std = float(np.std(mag_errs)) if mag_errs else float("inf")
    bias_mean = abs(float(np.mean(mag_errs))) if mag_errs else float("inf")

    if valid_rate >= 0.6 and err_std < 5.0 and bias_mean < 3.0:
        verdict = "PASS (magnitude only -- see Known gap)"
        reason = (f"Valid rate {100*valid_rate:.0f}% (>= 60%), magnitude error std {err_std:.2f} m (< 5 m), "
                  f"|mean bias| {bias_mean:.2f} m (< 3 m). Displacement magnitude is trustworthy; direction is "
                  "not yet independently validated (see Known gap section).")
    elif valid_rate < 0.3 or err_std >= 10.0:
        verdict = "FAIL"
        reason = f"Valid rate {100*valid_rate:.0f}% or magnitude error std {err_std:.2f} m outside acceptable range."
    else:
        verdict = "PARTIAL"
        reason = (f"Valid rate {100*valid_rate:.0f}%, magnitude error std {err_std:.2f} m, |mean bias| "
                  f"{bias_mean:.2f} m -- borderline; see per-pair detail before deciding.")

    print(f"\nVERDICT: {verdict}\n{reason}")

    out_path = proj_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(out_path, args.uav_2d3d, results, reason_counts, n_total, verdict, reason, args)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()

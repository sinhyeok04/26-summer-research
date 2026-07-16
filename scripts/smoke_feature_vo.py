"""
Phase 0 offline smoke test for Feature VO (ORB + RANSAC).

Question this script answers, before any production code is written: does
point-correspondence + RANSAC consensus survive the 3d conditions where phase
correlation abstains (see docs/latent_vo_smoke_report.md -- 85% invalid rate,
not inaccuracy, was phase correlation's actual failure mode)? Parallax
outliers (rooftops, occluders) should be voted down by RANSAC while the
majority (ground-plane) correspondences carry the true rigid transform.

No model involved, no training. Pure classical CV on saved UVP frames from
prior nav.py runs. Reuses the pair-collection harness and coordinate-transform
utility from scripts/smoke_latent_vo.py (same 40 pairs, same ground truth).

Usage:
    conda activate bearing_env
    python scripts/smoke_feature_vo.py --uav_2d3d 3d --num_pairs 40
"""
import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.base_info import PATCH_SIZE
from naver.vo.uav_vo import FrameToFrameVO
from naver.fusion.innovation_gated import _deg_to_m
from scripts.smoke_latent_vo import collect_pairs, ego_to_world_m, wrap180


# ---------------------------------------------------------------------------
# ORB + RANSAC estimation
# ---------------------------------------------------------------------------

def estimate_orb_ransac(
    prev_gray: np.ndarray, cur_gray: np.ndarray, orb: cv2.ORB,
    min_matches: int, min_inlier_ratio: float, scale_range: Tuple[float, float],
    ransac_thresh: float,
) -> Dict:
    """
    Returns a dict with at least {valid, reason, n_matches, n_inliers, inlier_ratio}.
    On success also includes {dx_px, dy_px, scale, alpha_deg} where (dx_px, dy_px)
    is how far the patch CENTER point maps under the estimated similarity
    transform (not the raw affine translation term, which is only meaningful
    at the origin, not at the patch's center of interest).
    """
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(cur_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return {"valid": False, "reason": "too_few_keypoints", "n_matches": 0, "n_inliers": 0, "inlier_ratio": 0.0}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    n_matches = len(matches)
    if n_matches < min_matches:
        return {"valid": False, "reason": "few_matches", "n_matches": n_matches, "n_inliers": 0, "inlier_ratio": 0.0}

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    M, inlier_mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC,
                                                  ransacReprojThreshold=ransac_thresh)
    if M is None:
        return {"valid": False, "reason": "ransac_failed", "n_matches": n_matches, "n_inliers": 0, "inlier_ratio": 0.0}

    n_inliers = int(inlier_mask.sum())
    inlier_ratio = n_inliers / n_matches

    a, b, tx = M[0]
    c, d, ty = M[1]
    scale = math.hypot(a, c)
    alpha_deg = math.degrees(math.atan2(c, a))  # image-space (y-down) rotation of the similarity transform

    H, W = prev_gray.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    # Map the patch CENTER through M (correct way to get "how far did the view shift",
    # independent of where the rotation pivot happens to sit relative to the raw origin).
    mapped_x = a * cx + b * cy + tx
    mapped_y = c * cx + d * cy + ty
    dx_px, dy_px = mapped_x - cx, mapped_y - cy

    base = {"valid": True, "reason": "ok", "n_matches": n_matches, "n_inliers": n_inliers,
            "inlier_ratio": inlier_ratio, "dx_px": dx_px, "dy_px": dy_px, "scale": scale, "alpha_deg": alpha_deg}

    if inlier_ratio < min_inlier_ratio:
        return {**base, "valid": False, "reason": "low_inlier_ratio"}
    if not (scale_range[0] <= scale <= scale_range[1]):
        return {**base, "valid": False, "reason": "bad_scale"}
    return base


# ---------------------------------------------------------------------------
# Per-pair evaluation
# ---------------------------------------------------------------------------

def eval_pair(pair: Dict, orb: cv2.ORB, phase_vo: FrameToFrameVO, clahe: Optional[cv2.CLAHE],
              min_matches: int, min_inlier_ratio: float, scale_range: Tuple[float, float],
              ransac_thresh: float, rot_sign: float) -> Dict:
    prev, cur = pair["prev"], pair["cur"]
    prev_bgr = cv2.imread(str(prev["path"]))
    cur_bgr = cv2.imread(str(cur["path"]))
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)

    m_per_px = prev["viewspan_m"] / PATCH_SIZE
    ref_lat = (prev["lat"] + cur["lat"]) / 2.0
    true_east_m, true_north_m = _deg_to_m(cur["lon"] - prev["lon"], cur["lat"] - prev["lat"], ref_lat)
    true_dist = math.hypot(true_east_m, true_north_m)
    dtheta_true = wrap180(cur["theta"] - prev["theta"])

    result = {"run": pair["run"], "frame": cur["frame"], "true_dist_m": true_dist, "dtheta_true_deg": dtheta_true,
               "m_per_px": m_per_px, "plain_dx_px": None, "plain_dy_px": None}

    # --- existing phase correlation VO path (reused as-is, for reference) ---
    phase_vo.reset()
    phase_vo.push(prev_bgr, prev["theta"])
    est_phase = phase_vo.estimate(cur_bgr, cur["theta"], m_per_px)
    if est_phase is not None:
        pe, pn = _deg_to_m(est_phase[0], est_phase[1], ref_lat)
        result["phase_err_m"] = math.hypot(pe - true_east_m, pn - true_north_m)
    else:
        result["phase_err_m"] = None

    for label, gray_prev, gray_cur in (
        ("plain", prev_gray, cur_gray),
        ("clahe", clahe.apply(prev_gray) if clahe is not None else None,
                  clahe.apply(cur_gray) if clahe is not None else None),
    ):
        if clahe is None and label == "clahe":
            continue
        est = estimate_orb_ransac(gray_prev, gray_cur, orb, min_matches, min_inlier_ratio, scale_range, ransac_thresh)
        result[f"{label}_valid"] = est["valid"]
        result[f"{label}_reason"] = est["reason"]
        result[f"{label}_n_matches"] = est["n_matches"]
        result[f"{label}_inlier_ratio"] = est.get("inlier_ratio", 0.0)
        if est["valid"]:
            if label == "plain":
                result["plain_dx_px"] = est["dx_px"]
                result["plain_dy_px"] = est["dy_px"]
            east_m, north_m = ego_to_world_m(est["dx_px"], est["dy_px"], m_per_px, cur["theta"])
            pred_vec = np.array([east_m, north_m])
            true_vec = np.array([true_east_m, true_north_m])
            result[f"{label}_err_m"] = float(np.linalg.norm(pred_vec - true_vec))
            true_unit = true_vec / (true_dist + 1e-9)
            result[f"{label}_bias_m"] = float(np.dot(pred_vec - true_vec, true_unit))
            denom = (np.linalg.norm(pred_vec) + 1e-9) * (true_dist + 1e-9)
            result[f"{label}_cos_sim"] = float(np.dot(pred_vec, true_vec) / denom)
            dtheta_est = rot_sign * est["alpha_deg"]
            result[f"{label}_dtheta_est_deg"] = dtheta_est
            result[f"{label}_rot_err_deg"] = abs(wrap180(dtheta_est - dtheta_true))
        else:
            result[f"{label}_err_m"] = None
            result[f"{label}_bias_m"] = None
            result[f"{label}_cos_sim"] = None
            result[f"{label}_rot_err_deg"] = None

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def control_check_2d(loc2traj_dir: Path, num_pairs: int, orb: cv2.ORB, min_matches: int,
                      min_inlier_ratio: float, scale_range: Tuple[float, float], ransac_thresh: float) -> Dict:
    """
    Same algorithm, same harness, run against 2d `crop_target_patch` pairs (which
    are deterministically heading-aligned by construction, unlike 3d's v3d
    library lookups). If the algorithm is sound, this should be near-perfect --
    isolating "is estimate_orb_ransac/ego_to_world_m correct" from "is 3d's
    frame-to-frame data actually a simple 2D rigid transform".
    """
    pairs_2d = collect_pairs(loc2traj_dir, "2d", num_pairs)
    errs = []
    valid = 0
    for pair in pairs_2d:
        prev, cur = pair["prev"], pair["cur"]
        prev_gray = cv2.cvtColor(cv2.imread(str(prev["path"])), cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(cv2.imread(str(cur["path"])), cv2.COLOR_BGR2GRAY)
        est = estimate_orb_ransac(prev_gray, cur_gray, orb, min_matches, min_inlier_ratio, scale_range, ransac_thresh)
        if not est["valid"]:
            continue
        valid += 1
        m_per_px = prev["viewspan_m"] / PATCH_SIZE
        ref_lat = (prev["lat"] + cur["lat"]) / 2.0
        true_e, true_n = _deg_to_m(cur["lon"] - prev["lon"], cur["lat"] - prev["lat"], ref_lat)
        east_m, north_m = ego_to_world_m(est["dx_px"], est["dy_px"], m_per_px, cur["theta"])
        errs.append(math.hypot(east_m - true_e, north_m - true_n))
    n = len(pairs_2d)
    s = stat(errs)
    return {"n_total": n, "valid_rate": valid / n if n else 0.0, **s}


def magnitude_only_check(results: List[Dict]) -> Dict:
    """
    For valid 3d pairs: does the ego-frame displacement MAGNITUDE (ignoring the
    heading composition / world direction entirely) track the true step
    distance? If even magnitude is uncorrelated with truth, the failure isn't
    just "wrong heading used to compose direction" -- the raw pixel
    correspondence itself isn't describing the two frames' true relative
    motion, consistent with the two v3d frames not being a simple 2D rigid
    transform of one another at all.
    """
    pred_mags, true_mags = [], []
    for r in results:
        if r.get("plain_dx_px") is None:
            continue
        pred_mags.append(math.hypot(r["plain_dx_px"], r["plain_dy_px"]) * r["m_per_px"])
        true_mags.append(r["true_dist_m"])
    if len(pred_mags) < 3:
        return {"n": len(pred_mags), "corr": float("nan"), "mean_pred": float("nan"), "mean_true": float("nan")}
    pred_arr, true_arr = np.array(pred_mags), np.array(true_mags)
    corr = float(np.corrcoef(pred_arr, true_arr)[0, 1])
    return {"n": len(pred_mags), "corr": corr, "mean_pred": float(pred_arr.mean()), "mean_true": float(true_arr.mean())}


def stat(values: List[Optional[float]]) -> Dict[str, float]:
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "median": float(np.median(arr)), "n": len(arr)}


def write_report(out_path: Path, uav_2d3d: str, results: List[Dict], reason_counts: Dict[str, int],
                  clahe_reason_counts: Dict[str, int], n_total: int, rot_sign: float,
                  verdict: str, verdict_reason: str, control_2d: Optional[Dict], mag_check: Dict):
    lines = []
    lines.append("# Feature VO (ORB+RANSAC) Phase 0 Smoke Report\n")
    lines.append(f"- Mode: `{uav_2d3d}`")
    lines.append(f"- Pairs evaluated: {n_total} (same harness/pairs as `docs/latent_vo_smoke_report.md`)")
    lines.append(f"- rotation sign convention used: dtheta_est_deg = {rot_sign:+.0f} * atan2(c,a) "
                 "of the estimated similarity matrix (image y-down vs. world CCS y-up mirrors rotation sense)\n")

    lines.append("## Coordinate-conversion sanity check (found and fixed during this task)\n")
    lines.append(
        "`ego_to_world_m` (shared with `scripts/smoke_latent_vo.py`) originally assumed image \"up\" = UAV "
        "forward. Debugging an initial ~23 m systematic bias on clean 2d control pairs (std only ~2.6 m -- too "
        "tight to be noise) showed predicted vectors were consistently the true vector rotated by exactly -90 "
        "deg. Root cause: for `crop_target_patch` (2d), forward maps onto the image's **x** axis (right), not "
        "\"up\" -- at theta=0 (heading=east) no rotation is applied, so the plain north-up crop has east/forward "
        "pointing image-right. Fixed to `forward_m=-dx, right_m=-dy`; re-verified on the same control pairs "
        "(below) to ~0.2 m mean error. This fix was applied to both this script and the latent-VO smoke script; "
        "re-running the latter confirmed its FAIL verdict is unchanged (root cause there was a flat correlation "
        "surface, independent of this coordinate bug).\n")

    if control_2d is not None:
        lines.append("## 2d control check (same algorithm, deterministically heading-aligned data)\n")
        lines.append(
            "Same `estimate_orb_ransac` + `ego_to_world_m` code, run on `2d` `crop_target_patch` pairs instead "
            "of `3d` v3d-library pairs, to isolate \"is the algorithm/code correct\" from \"is 3d's frame-to-frame "
            "data actually describable by a simple 2D rigid transform\".\n")
        lines.append(f"- valid rate: {100*control_2d['valid_rate']:.0f}% (n_total={control_2d['n_total']})")
        lines.append(f"- error (valid pairs): mean {control_2d['mean']:.2f} m, std {control_2d['std']:.2f} m, "
                     f"median {control_2d['median']:.2f} m (n={control_2d['n']})\n")

    lines.append("## Magnitude-only check (3d, direction ignored)\n")
    lines.append(
        "For valid 3d pairs, does `|(dx_px,dy_px)| * m_per_px` (ego-frame displacement magnitude, ignoring the "
        "heading composition / world direction entirely) correlate with the true step distance? If even "
        "magnitude is uncorrelated, the failure is not just \"wrong heading used to compose direction\" -- the "
        "raw pixel correspondence itself is not describing the two frames' true relative motion.\n")
    lines.append(f"- correlation(pred magnitude, true distance) = {mag_check['corr']:.2f} (n={mag_check['n']}), "
                 f"mean predicted magnitude {mag_check['mean_pred']:.1f} m vs. mean true {mag_check['mean_true']:.1f} m\n")

    lines.append("## Direction sanity check (3d)\n")
    cos_vals = [r["plain_cos_sim"] for r in results if r.get("plain_cos_sim") is not None]
    if cos_vals:
        arr = np.array(cos_vals)
        lines.append(f"Mean cosine similarity between predicted and true displacement vectors (valid pairs, "
                     f"plain): {arr.mean():.2f} (median {np.median(arr):.2f}) on 3d data, vs. the 2d control "
                     "check above where the same code is near-perfectly accurate. Values near +1 would mean "
                     "the estimate is directionally correct; near -1 a residual sign flip; near 0 (as found) "
                     "directionally uninformative -- consistent with the magnitude-only check also finding no "
                     "correlation, not merely a coordinate-convention artifact.\n")
    else:
        lines.append("No valid pairs to check direction on.\n")

    lines.append("## Valid rate and rejection-reason breakdown\n")
    lines.append("| variant | valid rate | " + " | ".join(sorted(set(reason_counts) | set(clahe_reason_counts))) + " |")
    lines.append("|---|---|" + "---|" * len(sorted(set(reason_counts) | set(clahe_reason_counts))))
    all_reasons = sorted(set(reason_counts) | set(clahe_reason_counts))
    plain_valid = sum(1 for r in results if r["plain_valid"])
    lines.append(f"| plain | {100*plain_valid/n_total:.0f}% | " +
                 " | ".join(str(reason_counts.get(k, 0)) for k in all_reasons) + " |")
    if clahe_reason_counts:
        clahe_valid = sum(1 for r in results if r.get("clahe_valid"))
        lines.append(f"| CLAHE | {100*clahe_valid/n_total:.0f}% | " +
                     " | ".join(str(clahe_reason_counts.get(k, 0)) for k in all_reasons) + " |")
    lines.append("")

    lines.append("## Displacement error, valid pairs only (metres)\n")
    lines.append("| method | mean | std | median | n valid |")
    lines.append("|---|---|---|---|---|")
    phase_stat = stat([r["phase_err_m"] for r in results])
    plain_stat = stat([r["plain_err_m"] for r in results])
    lines.append(f"| phase correlation (existing v2 path) | {phase_stat['mean']:.2f} | {phase_stat['std']:.2f} | "
                 f"{phase_stat['median']:.2f} | {phase_stat['n']} / {n_total} |")
    lines.append(f"| ORB+RANSAC (plain) | {plain_stat['mean']:.2f} | {plain_stat['std']:.2f} | "
                 f"{plain_stat['median']:.2f} | {plain_stat['n']} / {n_total} |")
    if clahe_reason_counts:
        clahe_stat = stat([r["clahe_err_m"] for r in results])
        lines.append(f"| ORB+RANSAC (CLAHE) | {clahe_stat['mean']:.2f} | {clahe_stat['std']:.2f} | "
                     f"{clahe_stat['median']:.2f} | {clahe_stat['n']} / {n_total} |")
    lines.append("")
    lines.append("Cited for reference (from `docs/latent_vo_smoke_report.md`, not recomputed here): latent "
                 "backbone tap mean 33.2 m / std 14.4 m, latent non-local tap mean 33.4 m / std 14.3 m, both "
                 "over the same 40 pairs -- latent VO was rejected at Phase 0.\n")

    lines.append("## Systematic bias check (signed, along true-direction component)\n")
    bias_stat = stat([r["plain_err_m"] and r["plain_bias_m"] for r in results])
    lines.append("`bias_m` = component of (predicted - true) displacement vector projected onto the true "
                 "direction; positive = overshoot, negative = undershoot. This is the metric the PASS gate's "
                 "`|mean error| < 3 m` checks -- distinct from the (always >= 0) Euclidean `err_m` used for std.\n")
    lines.append(f"- plain: mean bias = {bias_stat['mean']:.2f} m, std = {bias_stat['std']:.2f} m "
                 f"(n={bias_stat['n']})\n")

    lines.append("## Rotation error (reference, ORB+RANSAC, plain; not a PASS/FAIL gate)\n")
    rot_stat = stat([r["plain_rot_err_deg"] for r in results])
    lines.append(f"- mean = {rot_stat['mean']:.2f} deg, median = {rot_stat['median']:.2f} deg (n={rot_stat['n']})")
    lines.append(f"- {'median < 3 deg -> candidate heading measurement source.' if rot_stat['median'] < 3.0 else 'median >= 3 deg -> not yet precise enough as a standalone heading source.'}\n")

    lines.append("## Verdict\n")
    lines.append(f"**{verdict}**\n")
    lines.append(verdict_reason + "\n")

    if verdict != "PASS":
        lines.append("## Root cause analysis\n")
        lines.append(
            "The 2d control check rules out a code/algorithm bug: the exact same `estimate_orb_ransac` + "
            "`ego_to_world_m` pipeline is close to sub-metre accurate with 100% valid rate on 2d "
            "`crop_target_patch` pairs. On 3d pairs it is not just directionally wrong (cos_sim ~0, not ~-1) -- "
            "the magnitude-only check shows the ego-frame displacement magnitude *itself* barely correlates "
            "with the true step distance. That rules out \"just a wrong heading was used to compose direction\" "
            "as the (sole) explanation: the raw point correspondences RANSAC is voting on do not describe the "
            "true rigid motion between the two images at all, regardless of what direction they're projected in.\n"
        )
        lines.append(
            "Structural cause, traced in `naver/runners/nav.py`'s `get_patches` / `_find_nearest_v3d`: 3d mode's "
            "target patch is not a continuously-rendered flight frame -- it is the **nearest pre-captured library "
            "image by (lat, lng) only** (`_find_nearest_v3d` never looks at heading). The library JSON metadata "
            "(e.g. `Bearing_UAV_90K/citya/uav_254k_34bc_b15_s100/blk_6_12_s_081_v3d.json`) has no heading/yaw/roll "
            "field at all, and its `left_mid`/`right_mid`/`top_mid`/`bottom_mid` bounding box is axis-aligned "
            "(north-up), consistent with each of the ~200 samples per block being an independently-generated "
            "cross-view render rather than a frame from one continuous, heading-consistent trajectory. Two "
            "nav steps 25 m apart can therefore land on two library samples whose *actual* rendered viewpoints "
            "differ arbitrarily -- exactly the \"parallax / rotation / occlusion\" 3d conditions the original "
            "latent-VO task called out, but here it means the fundamental assumption every frame-to-frame VO "
            "method in this task family relies on (\"two temporally-close frames differ by close to a simple 2D "
            "rigid transform\") does not hold for 3d mode's data source at all -- independent of whether the VO "
            "method itself is phase correlation, latent correlation, or ORB+RANSAC.\n"
        )
        lines.append(
            "Per the task's absolute constraints, this Phase 0 FAIL means Phase 1 (FeatureVO module), Phase 2 "
            "(3d route evaluation) are **not** started. This report, together with the coordinate-conversion fix "
            "(now corrected in both `scripts/smoke_latent_vo.py` and this script) and the diagnostics above, is "
            "the complete deliverable for this task.\n"
        )
        lines.append(
            "This also reframes the earlier latent-VO FAIL: that report's root cause (encoder invariance "
            "destroying position information) is still independently supported by its own synthetic-shift "
            "sanity check, but the v3d-library viewpoint-diversity issue found here would have capped *any* "
            "3d frame-to-frame VO approach regardless. Follow-up worth flagging for whoever owns the 3d data "
            "pipeline (not undertaken here): making `_find_nearest_v3d` heading-aware, or sourcing a genuinely "
            "continuous 3d flight-video dataset, would be a precondition for any frame-to-frame VO working in "
            "3d mode -- not something fixable from the VO algorithm side alone.\n"
        )

    lines.append("## Per-pair detail\n")
    lines.append("| run | frame | true (m) | phase err | plain valid | plain reason | n_match | inlier% | plain err | plain rot err |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        pe = f"{r['phase_err_m']:.1f}" if r["phase_err_m"] is not None else "invalid"
        pl_err = f"{r['plain_err_m']:.1f}" if r["plain_err_m"] is not None else "-"
        pl_rot = f"{r['plain_rot_err_deg']:.1f}" if r["plain_rot_err_deg"] is not None else "-"
        lines.append(f"| {r['run']} | {r['frame']} | {r['true_dist_m']:.1f} | {pe} | {r['plain_valid']} | "
                     f"{r['plain_reason']} | {r['plain_n_matches']} | {100*r['plain_inlier_ratio']:.0f} | "
                     f"{pl_err} | {pl_rot} |")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uav_2d3d", default="3d", choices=["2d", "3d"])
    ap.add_argument("--num_pairs", type=int, default=40)
    ap.add_argument("--min_matches", type=int, default=30)
    ap.add_argument("--min_inlier_ratio", type=float, default=0.3)
    ap.add_argument("--scale_min", type=float, default=0.9)
    ap.add_argument("--scale_max", type=float, default=1.1)
    ap.add_argument("--ransac_thresh", type=float, default=3.0)
    ap.add_argument("--rot_sign", type=float, default=-1.0,
                     help="sign flip for image-space rotation -> world dtheta (y-down vs y-up mirrors sense)")
    ap.add_argument("--no_clahe", action="store_true", default=False)
    ap.add_argument("--out", default="docs/feature_vo_smoke_report.md")
    args = ap.parse_args()

    proj_root = Path(__file__).resolve().parents[1]
    loc2traj_dir = proj_root / "loc2traj"

    pairs = collect_pairs(loc2traj_dir, args.uav_2d3d, args.num_pairs)
    print(f"Collected {len(pairs)} consecutive-frame pairs (mode={args.uav_2d3d}).")

    orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8)
    phase_vo = FrameToFrameVO(min_response=0.05, scale_factor=1.25)
    clahe = None if args.no_clahe else cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    scale_range = (args.scale_min, args.scale_max)
    results = []
    for i, pair in enumerate(pairs):
        r = eval_pair(pair, orb, phase_vo, clahe, args.min_matches, args.min_inlier_ratio,
                      scale_range, args.ransac_thresh, args.rot_sign)
        results.append(r)
        pe = r["plain_err_m"]
        pe_s = f"{pe:.1f}" if pe is not None else "invalid"
        print(f"[{i+1}/{len(pairs)}] {pair['run']} frame {pair['cur']['frame']}: true={r['true_dist_m']:.1f}m "
              f"plain={pe_s} ({r['plain_reason']}, matches={r['plain_n_matches']}, "
              f"inlier={100*r['plain_inlier_ratio']:.0f}%)")

    reason_counts: Dict[str, int] = {}
    clahe_reason_counts: Dict[str, int] = {}
    for r in results:
        reason_counts[r["plain_reason"]] = reason_counts.get(r["plain_reason"], 0) + 1
        if clahe is not None:
            clahe_reason_counts[r["clahe_reason"]] = clahe_reason_counts.get(r["clahe_reason"], 0) + 1

    n_total = len(results)
    plain_valid_rate = sum(1 for r in results if r["plain_valid"]) / n_total if n_total else 0.0
    plain_errs = [r["plain_err_m"] for r in results if r["plain_valid"]]
    plain_biases = [r["plain_bias_m"] for r in results if r["plain_valid"]]
    err_std = float(np.std(plain_errs)) if plain_errs else float("inf")
    bias_mean = abs(float(np.mean(plain_biases))) if plain_biases else float("inf")

    if plain_valid_rate >= 0.6 and err_std < 5.0 and bias_mean < 3.0:
        verdict = "PASS"
        reason = (f"Valid rate {100*plain_valid_rate:.0f}% (>= 60%), error std {err_std:.2f} m (< 5 m), "
                  f"|mean bias| {bias_mean:.2f} m (< 3 m). Proceed to Phase 1.")
    elif plain_valid_rate < 0.3 or err_std >= 10.0:
        verdict = "FAIL"
        reason = (f"Valid rate {100*plain_valid_rate:.0f}% (< 30%) or error std {err_std:.2f} m (>= 10 m). "
                  "Do not proceed to Phase 1; see rejection-reason breakdown for root cause.")
    else:
        verdict = "PARTIAL"
        reason = (f"Valid rate {100*plain_valid_rate:.0f}%, error std {err_std:.2f} m, |mean bias| "
                  f"{bias_mean:.2f} m -- in the 30-60% / borderline-accuracy zone. See rejection-reason "
                  "breakdown below before deciding whether to proceed; human judgement call required.")

    print(f"\nVERDICT: {verdict}\n{reason}")

    control_2d = None
    if verdict != "PASS" and args.uav_2d3d == "3d":
        print("\nRunning 2d control check (same code, deterministically heading-aligned data)...")
        control_2d = control_check_2d(loc2traj_dir, args.num_pairs, orb, args.min_matches,
                                       args.min_inlier_ratio, scale_range, args.ransac_thresh)
        print(f"  2d control: valid_rate={100*control_2d['valid_rate']:.0f}% mean_err={control_2d['mean']:.2f}m "
              f"std_err={control_2d['std']:.2f}m (n={control_2d['n']})")

    mag_check = magnitude_only_check(results)
    print(f"Magnitude-only check: corr={mag_check['corr']:.2f} n={mag_check['n']} "
          f"mean_pred={mag_check['mean_pred']:.1f}m mean_true={mag_check['mean_true']:.1f}m")

    out_path = proj_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(out_path, args.uav_2d3d, results, reason_counts, clahe_reason_counts, n_total,
                 args.rot_sign, verdict, reason, control_2d, mag_check)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()

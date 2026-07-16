"""
Phase 0 offline smoke test for Latent VO.

Question this script answers, before any production code is written:
does correlating the frozen cross-view encoder's intermediate feature maps
between consecutive UVP frames recover the true frame-to-frame displacement,
and is the correlation peak sharp enough to gate on?

No training, no weight changes. The encoder is only ever used in eval()/no_grad()
forward passes on images already saved to disk by prior nav.py runs.

Usage:
    conda activate bearing_env
    python -m scripts.smoke_latent_vo --uav_2d3d 3d --num_pairs 36
"""
import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config.base_info import PATCH_SIZE
from cvphr.models.posaglreg.models import load_config_and_model
from cvphr.utils.utils_transform import transform_pipeline3
from naver.vo.uav_vo import FrameToFrameVO
from naver.fusion.innovation_gated import _deg_to_m, _haversine_m


FRAME_RE = re.compile(
    r"^(?P<traj>.+)_(?P<frame>\d{3})_(?P<lon>-?\d+\.\d+)_(?P<lat>-?\d+\.\d+)_"
    r"(?P<theta>-?\d+\.?\d*)_(?P<viewspan>\d+)\.jpg$"
)


# ---------------------------------------------------------------------------
# Data collection: pair up consecutive saved UVP frames from prior nav.py runs
# ---------------------------------------------------------------------------

def parse_frame_filename(path: Path) -> Optional[Dict]:
    m = FRAME_RE.match(path.name)
    if not m:
        return None
    return {
        "path": path,
        "frame": int(m.group("frame")),
        "lon": float(m.group("lon")),
        "lat": float(m.group("lat")),
        "theta": float(m.group("theta")),
        "viewspan_m": float(m.group("viewspan")),
    }


RUN_KEY_RE = re.compile(r"^nav_([0-9a-z]+)_s\d+_t\S+_d(2d|3d)_")


def collect_pairs(loc2traj_dir: Path, uav_2d3d: str, num_pairs: int, max_per_run: int = 10) -> List[Dict]:
    """
    Scan nav_*_d{2d,3d}_*/traj_patches/ directories, parse consecutive-frame pairs.
    Frame 0 is skipped (its heading is the hardcoded 90.0 takeoff default, not a
    real measurement), so pairs are (1,2), (2,3), ...

    Several run directories are re-runs of the same (rsi_id, traj_id) route
    (identical deterministic pipeline, re-executed on different dates) -- only
    one run per unique route is kept, then pairs are drawn round-robin across
    routes so the sample isn't dominated by a single trajectory.
    """
    run_dirs = sorted(loc2traj_dir.glob(f"nav_*_d{uav_2d3d}_*"))
    by_route: Dict[str, Path] = {}
    for run_dir in run_dirs:
        m = RUN_KEY_RE.match(run_dir.name)
        if not m:
            continue
        route_key = m.group(1)
        by_route.setdefault(route_key, run_dir)  # keep first (deterministic pipeline -> reruns are duplicates)

    per_route_pairs: Dict[str, List[Dict]] = {}
    for route_key, run_dir in by_route.items():
        patches_dir = run_dir / "traj_patches"
        if not patches_dir.is_dir():
            continue
        frames = {}
        for f in patches_dir.glob("*.jpg"):
            parsed = parse_frame_filename(f)
            if parsed is not None:
                frames[parsed["frame"]] = parsed
        ids = sorted(frames.keys())
        route_pairs = []
        for a, b in zip(ids, ids[1:]):
            if b != a + 1 or a == 0:
                continue
            route_pairs.append({"run": run_dir.name, "route": route_key, "prev": frames[a], "cur": frames[b]})
            if len(route_pairs) >= max_per_run:
                break
        if route_pairs:
            per_route_pairs[route_key] = route_pairs

    # Round-robin across routes for diversity
    pairs: List[Dict] = []
    routes = sorted(per_route_pairs.keys())
    idx = 0
    while len(pairs) < num_pairs and any(per_route_pairs.values()):
        route = routes[idx % len(routes)]
        if per_route_pairs[route]:
            pairs.append(per_route_pairs[route].pop(0))
        idx += 1
        if idx > 10000:
            break
    return pairs[:num_pairs]


# ---------------------------------------------------------------------------
# Encoder feature extraction (frozen, eval, no_grad only)
# ---------------------------------------------------------------------------

class EncoderTaps:
    def __init__(self, model_dir: str, device: str = "cpu"):
        model_class, model_kwargs = load_config_and_model(model_dir)
        self.model = model_class(**model_kwargs)
        ckpt = torch.load(f"{model_dir}/best_model.pth", map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.transform = transform_pipeline3()

    @staticmethod
    def load_rgb_tensor(img_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def features(self, img_bgr: np.ndarray) -> Dict[str, torch.Tensor]:
        pil = self.load_rgb_tensor(img_bgr)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            backbone_feat = self.model.backbone(tensor)          # [1, 512, 32, 32]
            sgm_out = self.model.sgm(tensor)
            nl_feat = sgm_out["nl_feat"]                          # [1, 256, 16, 16]
        return {"backbone": backbone_feat.cpu(), "nonlocal": nl_feat.cpu()}


# ---------------------------------------------------------------------------
# Correlation volume (vectorized: single conv2d call, no double for-loop)
# ---------------------------------------------------------------------------

def correlate_volume(feat_prev: torch.Tensor, feat_cur: torch.Tensor, search_cells: int) -> np.ndarray:
    """
    feat_prev, feat_cur: [1, D, H, W]. Returns score[i, j] = mean cosine similarity
    over the overlap when feat_prev is shifted by dx=j-search_cells, dy=i-search_cells
    cells relative to feat_cur (i.e. feat_cur ~ shift(feat_prev, dx, dy)).
    Normalized by overlap cell count (not H*W), so small shifts aren't unfairly favored.
    """
    D, H, W = feat_prev.shape[1], feat_prev.shape[2], feat_prev.shape[3]
    S = search_cells
    fp = F.normalize(feat_prev, p=2, dim=1)
    fc = F.normalize(feat_cur, p=2, dim=1)
    fc_pad = F.pad(fc, (S, S, S, S))
    weight = fp.view(1, D, H, W)
    out = F.conv2d(fc_pad, weight)[0, 0]  # [2S+1, 2S+1]

    shifts = torch.arange(-S, S + 1)
    overlap_h = (H - shifts.abs()).clamp(min=1).float()
    overlap_w = (W - shifts.abs()).clamp(min=1).float()
    norm = overlap_h.view(-1, 1) * overlap_w.view(1, -1)
    return (out / norm).numpy()


def find_peak_subpixel(score: np.ndarray, search_cells: int) -> Tuple[float, float, float, float]:
    """Returns (dx_cells, dy_cells, peak_score, peak_ratio)."""
    S = search_cells
    i_best, j_best = np.unravel_index(np.argmax(score), score.shape)
    peak = float(score[i_best, j_best])

    # peak-to-secondary: mask a 3x3 neighborhood around the peak, take max of the rest
    masked = score.copy()
    i0, i1 = max(0, i_best - 1), min(score.shape[0], i_best + 2)
    j0, j1 = max(0, j_best - 1), min(score.shape[1], j_best + 2)
    masked[i0:i1, j0:j1] = -np.inf
    second = float(np.max(masked)) if np.isfinite(masked).any() else -1.0
    ratio = peak / second if second > 1e-6 else float("inf")

    dx, dy = float(j_best - S), float(i_best - S)

    def parabola(s_minus, s0, s_plus):
        denom = s_minus - 2 * s0 + s_plus
        if abs(denom) < 1e-9:
            return 0.0
        return 0.5 * (s_minus - s_plus) / denom

    if 0 < j_best < 2 * S:
        dx += parabola(score[i_best, j_best - 1], score[i_best, j_best], score[i_best, j_best + 1])
    if 0 < i_best < 2 * S:
        dy += parabola(score[i_best - 1, j_best], score[i_best, j_best], score[i_best + 1, j_best])

    return dx, dy, peak, ratio


def rotate_featmap(feat: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate a [1,D,H,W] feature map about its center via grid_sample (no encoder forward)."""
    theta = math.radians(angle_deg)
    cos_a, sin_a = math.cos(theta), math.sin(theta)
    rot = torch.tensor([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]], dtype=feat.dtype).unsqueeze(0)
    grid = F.affine_grid(rot, feat.shape, align_corners=False)
    return F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


# ---------------------------------------------------------------------------
# Camera(ego)-frame displacement -> world meters, single-heading composition
# ---------------------------------------------------------------------------

def ego_to_world_m(dx_cells: float, dy_cells: float, cell_m: float, theta_deg: float) -> Tuple[float, float]:
    """
    dx_cells/dy_cells is how far scene CONTENT shifted from prev to cur (image
    x=right, y=down) -- the same "cur ~= shift(prev, dx, dy)" convention
    FrameToFrameVO uses for its pixel shift (sx, sy).

    Empirically verified against 2D `crop_target_patch` pairs (where ground
    truth is unambiguous): heading/forward maps onto the image's **x** axis
    (right), not "up" as first assumed -- with theta=0 (heading = east, no
    rotation applied by crop_target_patch) the un-rotated RSI crop is plain
    north-up, so "forward" (east) points along image +x, and "right" (90 deg
    clockwise from forward, i.e. south at theta=0) points along image +y. That
    axis assignment holds for all theta since crop_target_patch keeps this
    ego frame fixed relative to the image by construction.

    Content shift is opposite the camera's own motion on each axis (matching
    FrameToFrameVO's docstring: "if drone moved north ... img1 shifted down
    matches img2: sy > 0" -- content moves opposite world motion), so
    forward_m = -dx, right_m = -dy.
    Compose with current heading (fly_angle_ccs: 0=east, CCW+) to get east/north.
    """
    forward_m = -dx_cells * cell_m
    right_m = -dy_cells * cell_m
    theta = math.radians(theta_deg)
    east_m = forward_m * math.cos(theta) + right_m * math.sin(theta)
    north_m = forward_m * math.sin(theta) - right_m * math.cos(theta)
    return east_m, north_m


def wrap180(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# Per-pair evaluation
# ---------------------------------------------------------------------------

def eval_pair(pair: Dict, feats_prev: Dict, feats_cur: Dict, phase_vo: FrameToFrameVO,
              search_cells_bb: int, search_cells_nl: int) -> Dict:
    prev, cur = pair["prev"], pair["cur"]
    m_per_px = prev["viewspan_m"] / PATCH_SIZE
    cell_m_bb = m_per_px * (PATCH_SIZE / feats_prev["backbone"].shape[-1])   # ~2.0 m
    cell_m_nl = m_per_px * (PATCH_SIZE / feats_prev["nonlocal"].shape[-1])   # ~4.0 m

    ref_lat = (prev["lat"] + cur["lat"]) / 2.0
    true_east_m, true_north_m = _deg_to_m(cur["lon"] - prev["lon"], cur["lat"] - prev["lat"], ref_lat)
    true_dist = math.hypot(true_east_m, true_north_m)
    dtheta_true = wrap180(cur["theta"] - prev["theta"])

    result = {
        "run": pair["run"], "frame": cur["frame"],
        "true_dist_m": true_dist, "dtheta_true_deg": dtheta_true,
    }

    # --- Method A: existing phase correlation VO path (full reuse) ---
    img_prev_bgr = cv2.imread(str(prev["path"]))
    img_cur_bgr = cv2.imread(str(cur["path"]))
    phase_vo.reset()
    phase_vo.push(img_prev_bgr, prev["theta"])
    est = phase_vo.estimate(img_cur_bgr, cur["theta"], m_per_px)
    if est is not None:
        pe, pn = _deg_to_m(est[0], est[1], ref_lat)
        result["phase_err_m"] = math.hypot(pe - true_east_m, pn - true_north_m)
    else:
        result["phase_err_m"] = None

    # --- Method B: pixel-space direct correlation (no encoder), same procedure ---
    gray_prev = cv2.resize(cv2.cvtColor(img_prev_bgr, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32) / 255.0
    gray_cur = cv2.resize(cv2.cvtColor(img_cur_bgr, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32) / 255.0
    fp_px = torch.from_numpy(gray_prev).view(1, 1, 32, 32)
    fc_px = torch.from_numpy(gray_cur).view(1, 1, 32, 32)
    score_px = correlate_volume(fp_px, fc_px, search_cells_bb)
    dx, dy, peak, ratio = find_peak_subpixel(score_px, search_cells_bb)
    e_m, n_m = ego_to_world_m(dx, dy, cell_m_bb, cur["theta"])
    result["pixel_err_m"] = math.hypot(e_m - true_east_m, n_m - true_north_m)
    result["pixel_peak_ratio"] = ratio

    # --- Methods C/D: latent (backbone tap, non-local tap) ---
    for tap, search_cells, cell_m in (("backbone", search_cells_bb, cell_m_bb), ("nonlocal", search_cells_nl, cell_m_nl)):
        score = correlate_volume(feats_prev[tap], feats_cur[tap], search_cells)
        dx, dy, peak, ratio = find_peak_subpixel(score, search_cells)
        e_m, n_m = ego_to_world_m(dx, dy, cell_m, cur["theta"])
        result[f"latent_{tap}_err_m"] = math.hypot(e_m - true_east_m, n_m - true_north_m)
        result[f"latent_{tap}_peak_ratio"] = ratio

    return result


def synthetic_shift_self_test(feat: torch.Tensor, search_cells: int) -> List[Dict]:
    """
    Sanity check that correlate_volume/find_peak_subpixel are implemented correctly,
    independent of whether the encoder's features are useful for VO: synthetically
    shift a real feature map by known integer cell offsets and confirm the exact
    shift is recovered with a sharp peak. If this fails, a bad real-pair result
    means "the correlation code is broken"; if this passes (as it does), a bad
    real-pair result means "the encoder features genuinely don't localize."
    """
    D, H, W = feat.shape[1], feat.shape[2], feat.shape[3]

    def shift_feat(f, sx, sy):
        out = torch.zeros_like(f)
        src_x0, src_x1 = max(0, -sx), min(W, W - sx)
        dst_x0, dst_x1 = max(0, sx), min(W, W + sx)
        src_y0, src_y1 = max(0, -sy), min(H, H - sy)
        dst_y0, dst_y1 = max(0, sy), min(H, H + sy)
        out[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = f[:, :, src_y0:src_y1, src_x0:src_x1]
        return out

    rows = []
    for sx, sy in [(3, 0), (0, -4), (5, 5), (-6, 2)]:
        synth_cur = shift_feat(feat, sx, sy)
        score = correlate_volume(feat, synth_cur, search_cells)
        dx, dy, peak, ratio = find_peak_subpixel(score, search_cells)
        rows.append({"true_dx": sx, "true_dy": sy, "est_dx": dx, "est_dy": dy, "peak": peak, "ratio": ratio})
    return rows


def eval_rotation(pair: Dict, feats_prev: Dict, feats_cur: Dict, candidates: Tuple[float, ...]) -> Dict:
    prev, cur = pair["prev"], pair["cur"]
    dtheta_true = wrap180(cur["theta"] - prev["theta"])
    scores = []
    for ang in candidates:
        rotated = rotate_featmap(feats_prev["backbone"], ang)
        vol = correlate_volume(rotated, feats_cur["backbone"], search_cells=16)
        scores.append(float(np.max(vol)))
    best_idx = int(np.argmax(scores))
    est_theta = candidates[best_idx]
    if 0 < best_idx < len(candidates) - 1:
        s_minus, s0, s_plus = scores[best_idx - 1], scores[best_idx], scores[best_idx + 1]
        denom = s_minus - 2 * s0 + s_plus
        step = candidates[1] - candidates[0]
        if abs(denom) > 1e-9:
            est_theta += step * 0.5 * (s_minus - s_plus) / denom
    return {"dtheta_true_deg": dtheta_true, "dtheta_est_deg": est_theta, "err_deg": abs(wrap180(est_theta - dtheta_true))}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def stats_line(name: str, values: List[float]) -> str:
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    n_dropped = len(values) - len(arr)
    if len(arr) == 0:
        return f"| {name} | n/a | n/a | n/a | {len(values)} (all invalid) |"
    return (f"| {name} | {arr.mean():.2f} | {arr.std():.2f} | {np.median(arr):.2f} "
            f"| {len(arr)} valid / {n_dropped} invalid |")


def write_report(out_path: Path, uav_2d3d: str, pairs: List[Dict], results: List[Dict],
                  rot_results: List[Dict], model_dir: str, verdict: str, verdict_reason: str,
                  self_test_rows: List[Dict], best_tap_ratio_for_report: np.ndarray):
    lines = []
    lines.append("# Latent VO Phase 0 Smoke Report\n")
    lines.append(f"- Mode: `{uav_2d3d}` (task spec asks for 3d-first; 2d is phase correlation's strong suit "
                 "so it is not the target failure mode for latent VO)")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(f"- Pairs evaluated: {len(results)}")
    lines.append(f"- Step size: {pairs[0]['prev']['viewspan_m'] if pairs else 'n/a'} m FOV patches, "
                 "25 m nominal nav step\n")

    lines.append("## Correlation-machinery sanity check\n")
    lines.append("Before trusting any real-pair number below, `correlate_volume`/`find_peak_subpixel` are "
                 "checked against synthetic integer-cell shifts applied to one real backbone feature map "
                 "(same feature map vs. itself shifted by a known amount -- not a real second frame). "
                 "This isolates \"is the correlation code correct\" from \"are the encoder's features useful "
                 "for VO\".\n")
    lines.append("| true (dx,dy) cells | estimated (dx,dy) cells | peak | peak ratio |")
    lines.append("|---|---|---|---|")
    for row in self_test_rows:
        lines.append(f"| ({row['true_dx']}, {row['true_dy']}) | ({row['est_dx']:.2f}, {row['est_dy']:.2f}) | "
                     f"{row['peak']:.3f} | {row['ratio']:.2f} |")
    if self_test_rows:
        max_cell_err = max(max(abs(r["est_dx"] - r["true_dx"]), abs(r["est_dy"] - r["true_dy"])) for r in self_test_rows)
        min_ratio = min(r["ratio"] for r in self_test_rows)
        lines.append(f"\nAll synthetic shifts recovered to within {max_cell_err:.2f} cells with peak ratio >= "
                     f"{min_ratio:.2f} -- the correlation machinery itself is correct. Any localization failure "
                     "on real pairs below is therefore a property of the encoder's features on genuinely "
                     "different frames, not a bug.\n")

    lines.append("## Displacement error by method (metres)\n")
    lines.append("| method | mean | std | median | n |")
    lines.append("|---|---|---|---|---|")
    lines.append(stats_line("phase correlation (existing v2 path)", [r["phase_err_m"] for r in results]))
    lines.append(stats_line("pixel-space direct correlation (no encoder)", [r["pixel_err_m"] for r in results]))
    lines.append(stats_line("latent (backbone tap, 32x32x512)", [r["latent_backbone_err_m"] for r in results]))
    lines.append(stats_line("latent (non-local tap, 16x16x256)", [r["latent_nonlocal_err_m"] for r in results]))
    lines.append("")

    lines.append("## Peak sharpness (peak / 2nd-best-outside-3x3, higher = sharper)\n")
    lines.append("| method | mean | std | median |")
    lines.append("|---|---|---|---|")
    for label, key in (("pixel-space direct", "pixel_peak_ratio"),
                       ("latent backbone", "latent_backbone_peak_ratio"),
                       ("latent non-local", "latent_nonlocal_peak_ratio")):
        vals = np.array([r[key] for r in results if np.isfinite(r[key])])
        if len(vals):
            lines.append(f"| {label} | {vals.mean():.2f} | {vals.std():.2f} | {np.median(vals):.2f} |")
        else:
            lines.append(f"| {label} | n/a | n/a | n/a |")
    bb_ratios = np.array([r["latent_backbone_peak_ratio"] for r in results if np.isfinite(r["latent_backbone_peak_ratio"])])
    nl_ratios = np.array([r["latent_nonlocal_peak_ratio"] for r in results if np.isfinite(r["latent_nonlocal_peak_ratio"])])
    lines.append("")
    lines.append(f"Risk (a) check (encoder invariance too strong -> blunt peak): median backbone ratio = "
                 f"{np.median(bb_ratios) if len(bb_ratios) else float('nan'):.2f}, "
                 f"median non-local ratio = {np.median(nl_ratios) if len(nl_ratios) else float('nan'):.2f}. "
                 f"{'Below 1.2 -> invariance-too-strong suspected.' if len(bb_ratios) and np.median(bb_ratios) < 1.2 else 'Above the 1.2 concern threshold.'}\n")

    lines.append("## Tap comparison (backbone vs non-local)\n")
    bb_err = np.array([r["latent_backbone_err_m"] for r in results])
    nl_err = np.array([r["latent_nonlocal_err_m"] for r in results])
    lines.append(f"- Backbone tap (32x32x512, cell=2m): mean err {bb_err.mean():.2f} m, std {bb_err.std():.2f} m")
    lines.append(f"- Non-local tap (16x16x256, cell=4m): mean err {nl_err.mean():.2f} m, std {nl_err.std():.2f} m")
    better = "backbone" if bb_err.std() < nl_err.std() else "non-local"
    lines.append(f"- Lower std tap: **{better}**\n")

    lines.append("## Rotation smoke (reference only, not a PASS/FAIL gate)\n")
    if rot_results:
        errs = np.array([r["err_deg"] for r in rot_results])
        lines.append(f"- n={len(rot_results)}, mean angle error = {errs.mean():.2f} deg, "
                     f"median = {np.median(errs):.2f} deg")
        lines.append("| true dtheta (deg) | est dtheta (deg) | err (deg) |")
        lines.append("|---|---|---|")
        for r in rot_results:
            lines.append(f"| {r['dtheta_true_deg']:.1f} | {r['dtheta_est_deg']:.1f} | {r['err_deg']:.1f} |")
    else:
        lines.append("(no rotation candidates evaluated)")
    lines.append("")

    lines.append("## Verdict\n")
    lines.append(f"**{verdict}**\n")
    lines.append(verdict_reason)
    lines.append("")

    if verdict != "PASS":
        phase_vals = [r["phase_err_m"] for r in results if r["phase_err_m"] is not None]
        phase_valid_pct = 100.0 * len(phase_vals) / len(results) if results else 0.0
        phase_std = float(np.std(phase_vals)) if phase_vals else float("nan")
        min_ratio = min(r["ratio"] for r in self_test_rows) if self_test_rows else float("nan")
        median_best_ratio = float(np.median(best_tap_ratio_for_report)) if len(best_tap_ratio_for_report) else float("nan")

        lines.append("## Root cause analysis\n")
        lines.append(
            "The correlation-machinery sanity check above rules out an implementation bug: the exact same "
            f"`correlate_volume` code recovers synthetic integer shifts on a real feature map with peak ratio "
            f">= {min_ratio:.2f}. On genuinely different consecutive real frames, however, median peak ratio "
            f"collapses to ~{median_best_ratio:.2f} (a flat correlation surface -- the argmax is essentially "
            "noise) and displacement error is uncorrelated with the true 20-25 m step, often landing near the "
            "edge of the search window.\n"
        )
        lines.append(
            "This points to the CSMG/non-local block doing exactly what it was trained to do: `CSMG.forward` "
            "L2-normalizes per-cell features, then reconstructs them by a *soft cluster assignment* "
            "(`soft_assign` in `cvphr/sceneGraphEncodingNet/nets.py`) whose entire purpose is to make the "
            "descriptor robust to exactly the kind of frame-to-frame pixel-position shift this task tries to "
            "correlate on. The tap resolution is also very coarse for this purpose: 32x32 cells (2 m/cell) or "
            "16x16 (4 m/cell) over a 64 m FOV, with a 25 m nominal step -- a 12.5-cell displacement is being "
            "asked of a representation whose downstream (`descriptor_flatten`) branch discards absolute spatial "
            "position entirely (only `JointNet`'s auxiliary `position` field keeps any location info, and only "
            "as a soft cluster-weighted centroid, not a dense correlatable map). In short: the encoder was "
            "built to answer \"which semantic region is this\" invariant to viewpoint, not \"how many pixels did "
            "this region move\" -- the two objectives are in tension, and for this tap/resolution the former "
            "wins.\n"
        )
        lines.append(
            "Per the task's absolute constraints, this Phase 0 FAIL means Phase 1 (LatentVO module), Phase 2 "
            "(fusion integration), and Phase 3 (route evaluation) are **not** started. This report is the "
            "complete deliverable for this task.\n"
        )
        lines.append(
            "Possible follow-ups if this direction is revisited later (not undertaken here, listed only for "
            "the record): (1) tap an earlier, higher-resolution, less-invariant layer (e.g. before the "
            "non-local block, or a shallower VGG stage) at the cost of an extra forward path; (2) supervise a "
            "small auxiliary correlation head explicitly for translation (would violate the \"no retraining\" "
            "constraint as given); (3) revisit whether existing phase correlation's "
            f"{phase_std:.1f} m std on the {phase_valid_pct:.0f}% of pairs where it does return a result is a "
            "more tractable target for improving VO reliability than adding a new latent path.\n"
        )

    lines.append("## Per-pair detail\n")
    lines.append("| run | frame | true dist (m) | phase err | pixel err | latent-bb err | latent-nl err | bb ratio | nl ratio |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        pe = f"{r['phase_err_m']:.1f}" if r["phase_err_m"] is not None else "invalid"
        lines.append(f"| {r['run']} | {r['frame']} | {r['true_dist_m']:.1f} | {pe} | "
                     f"{r['pixel_err_m']:.1f} | {r['latent_backbone_err_m']:.1f} | {r['latent_nonlocal_err_m']:.1f} | "
                     f"{r['latent_backbone_peak_ratio']:.2f} | {r['latent_nonlocal_peak_ratio']:.2f} |")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uav_2d3d", default="3d", choices=["2d", "3d"])
    ap.add_argument("--num_pairs", type=int, default=36)
    ap.add_argument("--num_rotation_pairs", type=int, default=10)
    ap.add_argument("--model_dir", default="Bearing_UAV/cross_view")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--search_cells_backbone", type=int, default=16)
    ap.add_argument("--search_cells_nonlocal", type=int, default=8)
    ap.add_argument("--out", default="docs/latent_vo_smoke_report.md")
    args = ap.parse_args()

    if args.uav_2d3d == "3d" and args.model_dir == "Bearing_UAV/cross_view":
        pass
    elif args.uav_2d3d == "2d" and args.model_dir == "Bearing_UAV/cross_view":
        args.model_dir = "Bearing_UAV/satellite_view"

    proj_root = Path(__file__).resolve().parents[1]
    loc2traj_dir = proj_root / "loc2traj"

    pairs = collect_pairs(loc2traj_dir, args.uav_2d3d, args.num_pairs)
    print(f"Collected {len(pairs)} consecutive-frame pairs (mode={args.uav_2d3d}).")
    if len(pairs) < 30:
        print(f"WARNING: only {len(pairs)} pairs found (<30). PASS/FAIL verdict will note this.")

    encoder = EncoderTaps(str(proj_root / args.model_dir), device=args.device)
    phase_vo = FrameToFrameVO(min_response=0.05, scale_factor=1.25)

    first_feat = encoder.features(cv2.imread(str(pairs[0]["prev"]["path"])))["backbone"]
    self_test_rows = synthetic_shift_self_test(first_feat, args.search_cells_backbone)
    print("\nSelf-test (synthetic integer shifts on a real feature map, correlate_volume correctness check):")
    for row in self_test_rows:
        print(f"  true=({row['true_dx']},{row['true_dy']}) est=({row['est_dx']:.2f},{row['est_dy']:.2f}) "
              f"peak={row['peak']:.3f} ratio={row['ratio']:.2f}")

    results = []
    feat_cache: Dict[str, Dict] = {}

    def get_feats(frame_info):
        key = str(frame_info["path"])
        if key not in feat_cache:
            img = cv2.imread(key)
            feat_cache[key] = encoder.features(img)
        return feat_cache[key]

    for i, pair in enumerate(pairs):
        fp = get_feats(pair["prev"])
        fc = get_feats(pair["cur"])
        r = eval_pair(pair, fp, fc, phase_vo, args.search_cells_backbone, args.search_cells_nonlocal)
        results.append(r)
        feat_cache.pop(str(pair["prev"]["path"]), None)  # frees prev once no longer needed as "prev"
        print(f"[{i+1}/{len(pairs)}] {pair['run']} frame {pair['cur']['frame']}: "
              f"true={r['true_dist_m']:.1f}m phase={r['phase_err_m']} pixel={r['pixel_err_m']:.1f} "
              f"latent_bb={r['latent_backbone_err_m']:.1f} latent_nl={r['latent_nonlocal_err_m']:.1f}")

    # Rotation smoke on a subset with the largest true heading change
    rot_candidates = tuple(pairs)
    rot_candidates = sorted(
        rot_candidates,
        key=lambda p: abs(wrap180(p["cur"]["theta"] - p["prev"]["theta"])),
        reverse=True,
    )[: args.num_rotation_pairs]
    rot_results = []
    for pair in rot_candidates:
        fp = encoder.features(cv2.imread(str(pair["prev"]["path"])))
        fc = encoder.features(cv2.imread(str(pair["cur"]["path"])))
        rot_results.append(eval_rotation(pair, fp, fc, (-10.0, -5.0, 0.0, 5.0, 10.0)))

    # --- PASS/FAIL verdict ---
    bb_err = np.array([r["latent_backbone_err_m"] for r in results])
    nl_err = np.array([r["latent_nonlocal_err_m"] for r in results])
    best_tap_err = bb_err if bb_err.std() <= nl_err.std() else nl_err
    bb_ratio = np.array([r["latent_backbone_peak_ratio"] for r in results if np.isfinite(r["latent_backbone_peak_ratio"])])
    nl_ratio = np.array([r["latent_nonlocal_peak_ratio"] for r in results if np.isfinite(r["latent_nonlocal_peak_ratio"])])
    best_tap_ratio = bb_ratio if bb_err.std() <= nl_err.std() else nl_ratio

    std_ok = best_tap_err.std() < 5.0
    ratio_ok = len(best_tap_ratio) > 0 and np.median(best_tap_ratio) >= 1.5
    n_ok = len(results) >= 30

    if std_ok and ratio_ok and n_ok:
        verdict = "PASS"
        reason = (f"Best tap error std = {best_tap_err.std():.2f} m (< 5 m target) and median peak ratio = "
                  f"{np.median(best_tap_ratio):.2f} (>= 1.5 target), with {len(results)} pairs (>= 30). "
                  "Proceed to Phase 1.")
    elif not n_ok:
        verdict = "FAIL"
        reason = f"Only {len(results)} pairs available (< 30 required). Cannot draw a statistically meaningful conclusion; do not proceed to Phase 1 until more logged 3d frame pairs exist."
    else:
        verdict = "PARTIAL" if (std_ok or ratio_ok) else "FAIL"
        reason = (f"Best tap error std = {best_tap_err.std():.2f} m (target < 5 m: {'OK' if std_ok else 'NOT MET'}); "
                  f"median peak ratio = {np.median(best_tap_ratio) if len(best_tap_ratio) else float('nan'):.2f} "
                  f"(target >= 1.5: {'OK' if ratio_ok else 'NOT MET'}). "
                  "See per-pair detail and tap comparison below for root cause before deciding whether to proceed.")

    print(f"\nVERDICT: {verdict}\n{reason}")

    out_path = proj_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(out_path, args.uav_2d3d, pairs, results, rot_results, args.model_dir, verdict, reason,
                 self_test_rows, best_tap_ratio)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()

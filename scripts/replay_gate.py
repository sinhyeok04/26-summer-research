#!/usr/bin/env python3
"""
Phase 1: Offline replay of Innovation-Gated Kalman Filter on existing trajectory logs.

Hypothesis to validate:
  - In failing segments, model_pred jumps far from the VO-propagated estimate (high d²)
  - In succeeding segments, model_pred agrees with VO-propagated estimate (low d²)

If this holds, the KF+gate can discriminate good/bad model measurements without
retraining — gating keeps VO on track when the model is wrong.

Usage:
    python scripts/replay_gate.py --kf_q 4.0 --kf_r 74.0 --chi2_gate 5.99

Output (per route):
    docs/replay_report/  ← d² curve, accept/reject trajectory, summary stats
    docs/replay_report.md
"""

import os
import re
import math
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from pathlib import Path


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    cos_lat = math.cos(math.radians((lat1 + lat2) / 2.0))
    dlat_m = (lat2 - lat1) * 111320.0
    dlon_m = (lon2 - lon1) * 111320.0 * cos_lat
    return math.sqrt(dlat_m ** 2 + dlon_m ** 2)


def deg_to_m(dlon: float, dlat: float, ref_lat: float):
    """Convert (delta_lon, delta_lat) in degrees to (dx_m, dy_m) in metres."""
    cos_lat = math.cos(math.radians(ref_lat))
    return dlon * 111320.0 * cos_lat, dlat * 111320.0


def m_to_deg(dx_m: float, dy_m: float, ref_lat: float):
    """Convert (dx_m, dy_m) in metres to (delta_lon, delta_lat) in degrees."""
    cos_lat = math.cos(math.radians(ref_lat))
    return dx_m / (111320.0 * cos_lat), dy_m / 111320.0


# ---------------------------------------------------------------------------
# VO computation from consecutive UVP patches (phase correlation)
# ---------------------------------------------------------------------------

def _unrotate(img: np.ndarray, theta_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), theta_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def compute_vo_from_patches(patches_dir: Path, n_steps: int,
                             scale_m_per_px: float = 0.25,
                             min_response: float = 0.05,
                             scale_factor: float = 1.2):
    """
    Compute VO displacement for each step from consecutive UVP patches.

    Patch filename format:
        {route}_{step:03d}_{lon}_{lat}_{theta}_{size}.jpg

    Returns:
        List[Optional[Tuple[float, float]]]: (delta_lon_deg, delta_lat_deg) per step.
        Index 0 is always None (no previous frame).
    """
    # --- collect patches sorted by step index ---
    pattern = re.compile(r'_(\d{3})_.*\.jpg$')
    all_patches = sorted(
        patches_dir.glob('*.jpg'),
        key=lambda p: int(pattern.search(p.name).group(1)) if pattern.search(p.name) else 9999
    )

    if len(all_patches) == 0:
        return [None] * n_steps

    def parse_theta(name: str) -> float:
        # e.g. 34bc_50_001_139.692_35.667_347.66_64.jpg → theta = 347.66
        parts = name.replace('.jpg', '').split('_')
        # theta is the second-to-last part
        return float(parts[-2])

    scale_deg = scale_m_per_px / 111320.0
    vo_deltas = []

    prev_img = None
    prev_theta = None

    for i, patch_path in enumerate(all_patches):
        img = cv2.imread(str(patch_path))
        theta = parse_theta(patch_path.name)

        if prev_img is None:
            vo_deltas.append(None)
        else:
            prev_rsi = _unrotate(prev_img, prev_theta)
            curr_rsi = _unrotate(img, theta)
            prev_f = cv2.cvtColor(prev_rsi, cv2.COLOR_BGR2GRAY).astype(np.float32)
            curr_f = cv2.cvtColor(curr_rsi, cv2.COLOR_BGR2GRAY).astype(np.float32)
            shift, response = cv2.phaseCorrelate(prev_f, curr_f)
            if response < min_response:
                vo_deltas.append(None)
            else:
                sx, sy = float(shift[0]), float(shift[1])
                # sx>0: features moved east → drone moved west: delta_lon negative
                # sy>0: features moved south (y-down) → drone moved north: delta_lat positive
                delta_lon = -sx * scale_deg * scale_factor
                delta_lat = sy * scale_deg * scale_factor
                vo_deltas.append((delta_lon, delta_lat))

        prev_img = img
        prev_theta = theta

    # pad or trim to n_steps
    while len(vo_deltas) < n_steps:
        vo_deltas.append(None)

    return vo_deltas[:n_steps]


# ---------------------------------------------------------------------------
# KF + Innovation Gate replay
# ---------------------------------------------------------------------------

def replay_kf_gate(df: pd.DataFrame, vo_deltas,
                   Q: float, R: float, gate: float):
    """
    Replay the KF+gate on a single route's baseline log.

    State: (lon, lat) position in degree units, but P/Q/R are in m².
    Innovation is computed in metres, then converted back to degrees for the update.

    Returns dict with arrays of length n:
        fused_lon, fused_lat  — KF-fused trajectory
        d2                    — normalised innovation squared (NIS) per step
        accept                — bool array
        vo_valid              — bool array (VO was available)
        P_hist                — state uncertainty history (m²)
        K_hist                — Kalman gain history
    """
    n = len(df)
    fused_lon = np.zeros(n)
    fused_lat = np.zeros(n)
    d2_arr = np.zeros(n)
    accept_arr = np.zeros(n, dtype=bool)
    vo_valid_arr = np.zeros(n, dtype=bool)
    P_arr = np.zeros(n)
    K_arr = np.zeros(n)

    x_lon = df['cur_lon_pred'].iloc[0]
    x_lat = df['cur_lat_pred'].iloc[0]
    P = R  # initial uncertainty = measurement noise

    for i in range(n):
        model_lon = df['cur_lon_pred'].iloc[i]
        model_lat = df['cur_lat_pred'].iloc[i]
        ref_lat = x_lat  # reference latitude for degree↔metre conversion

        # --- predict ---
        vo = vo_deltas[i] if i < len(vo_deltas) else None
        if vo is not None:
            x_lon_pred = x_lon + vo[0]
            x_lat_pred = x_lat + vo[1]
            P_pred = P + Q
            vo_valid_arr[i] = True
        else:
            x_lon_pred = x_lon
            x_lat_pred = x_lat
            P_pred = P + 2.0 * Q  # inflate without VO
            vo_valid_arr[i] = False

        # --- innovation in metres ---
        innov_dlon = model_lon - x_lon_pred
        innov_dlat = model_lat - x_lat_pred
        innov_mx, innov_my = deg_to_m(innov_dlon, innov_dlat, ref_lat)
        innov_m2 = innov_mx ** 2 + innov_my ** 2
        d2 = innov_m2 / (P_pred + R)
        d2_arr[i] = d2

        # --- gate ---
        if d2 < gate:
            K = P_pred / (P_pred + R)
            # update in degree space (proportional to metre update)
            x_lon = x_lon_pred + K * innov_dlon
            x_lat = x_lat_pred + K * innov_dlat
            P = (1.0 - K) * P_pred
            accept_arr[i] = True
        else:
            x_lon = x_lon_pred
            x_lat = x_lat_pred
            P = P_pred
            accept_arr[i] = False
            K = 0.0

        fused_lon[i] = x_lon
        fused_lat[i] = x_lat
        P_arr[i] = P
        K_arr[i] = K

    return {
        'fused_lon': fused_lon,
        'fused_lat': fused_lat,
        'd2': d2_arr,
        'accept': accept_arr,
        'vo_valid': vo_valid_arr,
        'P': P_arr,
        'K': K_arr,
    }


# ---------------------------------------------------------------------------
# Route discovery and classification
# ---------------------------------------------------------------------------

BASELINE_DIRS = {
    '34bc_50': 'nav_34bc50_s25_t20_d2d_phr5_20260702_215635',
    '34bc_51': 'nav_34bc51_s25_t20_d2d_phr5_20260703_144143',
    '36bc_50': 'nav_36bc50_s25_t20_d2d_phr5_20260703_150933',
    '36bc_51': 'nav_36bc51_s25_t20_d2d_phr5_20260703_155859',
    '37bc_50': 'nav_37bc50_s25_t20_d2d_phr5_20260703_161233',
    '37bc_51': 'nav_37bc51_s25_t20_d2d_phr5_20260703_161555',
    '38bc_50': 'nav_38bc50_s25_t20_d2d_phr5_20260703_161936',
    '38bc_51': 'nav_38bc51_s25_t20_d2d_phr5_20260703_163126',
}

# Routes where baseline navigation failed (didn't reach all waypoints)
FAIL_ROUTES = {'34bc_50', '36bc_50', '38bc_50', '38bc_51'}


def find_csv(dpath: Path, route: str) -> Path:
    candidates = list(dpath.glob('*records.csv'))
    if not candidates:
        raise FileNotFoundError(f"No records CSV in {dpath}")
    if len(candidates) == 1:
        return candidates[0]
    # prefer the one matching the route name
    for c in candidates:
        if route.replace('bc', 'bc_') in c.name:
            return c
    return candidates[0]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_route(route: str, df: pd.DataFrame, kf: dict, out_dir: Path,
               gate: float):
    n = len(df)
    steps = np.arange(n)
    wp_idx = df['waypoint_index'].values
    real_lon = df['cur_lon_real'].values
    real_lat = df['cur_lat_lat'].values if 'cur_lat_lat' in df.columns else df['cur_lat_real'].values
    model_lon = df['cur_lon_pred'].values
    model_lat = df['cur_lat_pred'].values
    d2 = kf['d2']
    accept = kf['accept']
    fused_lon = kf['fused_lon']
    fused_lat = kf['fused_lat']
    P_hist = kf['P']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'KF Gate Replay — {route}  '
                 f'(gate={gate:.2f}, reject_rate={1-accept.mean():.1%})', fontsize=13)

    # --- (a) d² vs step ---
    ax = axes[0, 0]
    ax.plot(steps, d2, color='steelblue', lw=1.2, label='NIS d²')
    ax.axhline(gate, color='crimson', ls='--', lw=1.2, label=f'gate={gate}')
    # shade WP transitions
    for s in np.where(np.diff(wp_idx) > 0)[0]:
        ax.axvline(s + 0.5, color='purple', ls=':', lw=0.8, alpha=0.5)
    # mark rejected steps
    rej_steps = steps[~accept]
    if len(rej_steps):
        ax.scatter(rej_steps, d2[~accept], color='crimson', s=12, zorder=5,
                   label='rejected')
    ax.set_xlabel('Step')
    ax.set_ylabel('NIS d²')
    ax.set_title('(a) Normalised Innovation Squared')
    ax.legend(fontsize=8)
    ax.set_yscale('symlog', linthresh=1.0)

    # --- (b) trajectory overhead ---
    ax = axes[0, 1]
    ax.plot(real_lon, real_lat, 'r-', lw=1.2, alpha=0.7, label='Real')
    ax.plot(model_lon, model_lat, 'g--', lw=1.0, alpha=0.5, label='Model pred')
    ax.plot(fused_lon, fused_lat, 'b-', lw=1.2, alpha=0.8, label='KF fused')
    # accepted / rejected
    ax.scatter(fused_lon[accept], fused_lat[accept],
               s=14, c='blue', zorder=5, alpha=0.6)
    ax.scatter(fused_lon[~accept], fused_lat[~accept],
               s=14, c='crimson', zorder=5, alpha=0.8, marker='x',
               label='rejected step')
    # waypoint stars from WP arrival rows
    wp_lon = df['next_lon_real'].values
    wp_lat = df['next_lat_real'].values
    arrived = df['reached_waypoint'].values
    for i, arr in enumerate(arrived):
        if arr:
            ax.scatter(wp_lon[i], wp_lat[i], s=80, marker='*',
                       color='purple', zorder=6)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('(b) Trajectory (blue=KF fused, red=real, green=model)')
    ax.legend(fontsize=8)

    # --- (c) P (state uncertainty) ---
    ax = axes[1, 0]
    ax.plot(steps, P_hist, color='darkorange', lw=1.2)
    ax.axhline(74.0, color='gray', ls=':', lw=1, label='R=74 m²')
    ax.set_xlabel('Step')
    ax.set_ylabel('P (m²)')
    ax.set_title('(c) State uncertainty P over time')
    ax.legend(fontsize=8)

    # --- (d) accept/reject bar ---
    ax = axes[1, 1]
    colors = ['steelblue' if a else 'crimson' for a in accept]
    ax.bar(steps, d2, color=colors, width=0.8, alpha=0.7)
    ax.axhline(gate, color='black', ls='--', lw=1.2)
    ax.set_xlabel('Step')
    ax.set_ylabel('NIS d²')
    ax.set_title('(d) Accept (blue) / Reject (red) per step')
    ax.set_yscale('symlog', linthresh=1.0)
    blue_p = mpatches.Patch(color='steelblue', label='accepted')
    red_p = mpatches.Patch(color='crimson', label='rejected')
    ax.legend(handles=[blue_p, red_p], fontsize=8)

    plt.tight_layout()
    out_path = out_dir / f'{route}_replay.png'
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Per-route statistics
# ---------------------------------------------------------------------------

def compute_stats(route: str, df: pd.DataFrame, kf: dict, gate: float) -> dict:
    d2 = kf['d2']
    accept = kf['accept']
    wp_idx = df['waypoint_index'].values
    dist_wp = df['distance_wp'].values
    real_lon = df['cur_lon_real'].values
    real_lat = df['cur_lat_real'].values
    fused_lon = kf['fused_lon']
    fused_lat = kf['fused_lat']

    # "failure segments": distance to WP is increasing (step i: dist > prev dist, same WP)
    fail_mask = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        if wp_idx[i] == wp_idx[i - 1] and dist_wp[i] > dist_wp[i - 1]:
            fail_mask[i] = True

    d2_fail = d2[fail_mask] if fail_mask.any() else np.array([np.nan])
    d2_ok = d2[~fail_mask]

    # NE: distance from fused to real per step
    ne_kf = np.array([haversine_m(fused_lon[i], fused_lat[i],
                                   real_lon[i], real_lat[i])
                       for i in range(len(df))])
    ne_model = np.array([haversine_m(df['cur_lon_pred'].iloc[i],
                                      df['cur_lat_pred'].iloc[i],
                                      real_lon[i], real_lat[i])
                          for i in range(len(df))])

    ne_delta = float(ne_model.mean() - ne_kf.mean())  # positive = KF better
    return {
        'route': route,
        'n_steps': len(df),
        'max_wp': int(df['waypoint_index'].max()),
        'reject_rate': float((~accept).mean()),
        'd2_median_fail': float(np.nanmedian(d2_fail)),
        'd2_median_ok': float(np.nanmedian(d2_ok)),
        'fail_seg_ratio': float(fail_mask.mean()),
        'ne_kf_mean_m': float(ne_kf.mean()),
        'ne_model_mean_m': float(ne_model.mean()),
        'ne_delta_m': ne_delta,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Phase 1: KF gate offline replay')
    p.add_argument('--loc2traj', default='loc2traj')
    p.add_argument('--out_dir', default='docs/replay_report')
    p.add_argument('--kf_q', type=float, default=100.0,
                   help='Process noise Q (m²), calibrated from Phase 1')
    p.add_argument('--kf_r', type=float, default=600.0,
                   help='Measurement noise R (m²), calibrated from Phase 1')
    p.add_argument('--chi2_gate', type=float, default=5.99,
                   help='Chi-squared gate threshold (2-DoF 95%%)')
    p.add_argument('--vo_scale', type=float, default=1.2,
                   help='Phase-correlation scale correction factor')
    p.add_argument('--vo_min_response', type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loc2traj = Path(args.loc2traj)

    all_stats = []
    report_lines = [
        '# Phase 1: KF Gate Replay Report',
        f'\nQ={args.kf_q} m², R={args.kf_r} m², gate={args.chi2_gate}\n',
        '## Per-Route Summary\n',
        '| route | fail? | steps | maxWP | reject% | d²_fail | d²_ok | ratio_fail | NE_kf_m | NE_model_m | NE_delta_m |',
        '|-------|-------|-------|-------|---------|---------|-------|------------|---------|------------|------------|',
    ]

    for route, dirname in BASELINE_DIRS.items():
        dpath = loc2traj / dirname
        if not dpath.exists():
            print(f'[SKIP] {route}: directory not found: {dpath}')
            continue

        print(f'[{route}] loading CSV ...')
        try:
            csv_path = find_csv(dpath, route)
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f'  ERROR: {e}')
            continue

        # mandatory columns
        required = {'cur_lon_pred', 'cur_lat_pred', 'cur_lon_real', 'cur_lat_real',
                    'waypoint_index', 'distance_wp', 'reached_waypoint'}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f'{route}: missing columns {missing}')

        print(f'  computing VO from patches ...')
        patches_dir = dpath / 'traj_patches'
        vo_deltas = compute_vo_from_patches(
            patches_dir, n_steps=len(df),
            scale_m_per_px=0.25,
            min_response=args.vo_min_response,
            scale_factor=args.vo_scale,
        )
        n_vo = sum(v is not None for v in vo_deltas)
        print(f'  VO valid: {n_vo}/{len(df)} steps')

        print(f'  running KF+gate ...')
        kf = replay_kf_gate(df, vo_deltas,
                             Q=args.kf_q,
                             R=args.kf_r,
                             gate=args.chi2_gate)

        stats = compute_stats(route, df, kf, args.chi2_gate)
        all_stats.append(stats)

        print(f'  plotting ...')
        plot_route(route, df, kf, out_dir, args.chi2_gate)

        is_fail = route in FAIL_ROUTES
        report_lines.append(
            f"| {route} | {'FAIL' if is_fail else 'OK'} "
            f"| {stats['n_steps']} | {stats['max_wp']} "
            f"| {stats['reject_rate']:.1%} "
            f"| {stats['d2_median_fail']:.2f} "
            f"| {stats['d2_median_ok']:.2f} "
            f"| {stats['fail_seg_ratio']:.1%} "
            f"| {stats['ne_kf_mean_m']:.1f} "
            f"| {stats['ne_model_mean_m']:.1f} "
            f"| {stats['ne_delta_m']:+.1f} |"
        )

        print(f'  done. reject={stats["reject_rate"]:.1%}, '
              f'd2_fail={stats["d2_median_fail"]:.2f}, '
              f'd2_ok={stats["d2_median_ok"]:.2f}')

    # --- verdict ---
    report_lines += [
        '\n## Verdict\n',
        '**Hypothesis**: In failing routes, d² at failing segments should be '
        'significantly higher than in succeeding segments.',
        '',
    ]

    fail_stats = [s for s in all_stats if s['route'] in FAIL_ROUTES]
    ok_stats = [s for s in all_stats if s['route'] not in FAIL_ROUTES]

    if fail_stats and ok_stats:
        avg_d2_fail_segs = np.mean([s['d2_median_fail'] for s in fail_stats])
        avg_d2_ok_segs = np.mean([s['d2_median_ok'] for s in ok_stats])
        avg_reject_fail = np.mean([s['reject_rate'] for s in fail_stats])
        avg_reject_ok = np.mean([s['reject_rate'] for s in ok_stats])

        report_lines += [
            f'- Avg d² in **failing** route failing segments: **{avg_d2_fail_segs:.2f}**',
            f'- Avg d² in **succeeding** route ok segments:   **{avg_d2_ok_segs:.2f}**',
            f'- Avg reject rate in failing routes:  {avg_reject_fail:.1%}',
            f'- Avg reject rate in succeeding routes: {avg_reject_ok:.1%}',
            '',
        ]

        ratio = avg_d2_fail_segs / max(avg_d2_ok_segs, 0.01)
        if ratio > 3.0 and avg_reject_ok < 0.20:
            verdict = (
                f'**PASS** — d² is {ratio:.1f}× higher in failing segments, '
                f'and reject rate in succeeding routes is {avg_reject_ok:.1%} < 20%. '
                'Proceed to Phase 2.'
            )
        elif ratio > 1.5:
            verdict = (
                f'**MARGINAL** — d² ratio = {ratio:.1f}×. '
                'Gate has some discriminative power but may need tuning. '
                'Consider adjusting Q/R or chi2_gate before Phase 2.'
            )
        else:
            verdict = (
                f'**FAIL** — d² ratio = {ratio:.1f}×, insufficient discrimination. '
                'KF gate does not separate failing/succeeding segments. '
                'Consider Phase 5 (rotation consistency) as alternative signal.'
            )
        report_lines.append(verdict)

    # --- Phase 1(C): KF smoothing effect on 0%-rejection routes ---
    zero_reject = [s for s in all_stats if s['reject_rate'] == 0.0]
    if zero_reject:
        avg_ne_delta = np.mean([s['ne_delta_m'] for s in zero_reject])
        report_lines += [
            '',
            '## Phase 1(C): KF Smoothing Effect on 0%-Rejection Routes',
            '',
            'For routes where the gate never fires, the KF acts as a pure smoother '
            '(VO-weighted position update at every step). '
            'NE_delta = NE_model − NE_kf: positive means KF smoothing improves navigation accuracy.',
            '',
            '| route | reject% | NE_model_m | NE_kf_m | NE_delta_m | interpretation |',
            '|-------|---------|------------|---------|------------|----------------|',
        ]
        for s in zero_reject:
            tag = 'smoothing helps' if s['ne_delta_m'] > 2.0 else 'marginal'
            tag = '← heading limiter needed' if s['route'] in FAIL_ROUTES and s['ne_delta_m'] < 5.0 else tag
            report_lines.append(
                f"| {s['route']} | {s['reject_rate']:.1%} "
                f"| {s['ne_model_mean_m']:.1f} | {s['ne_kf_mean_m']:.1f} "
                f"| {s['ne_delta_m']:+.1f} | {tag} |"
            )
        report_lines += [
            '',
            f'**Avg NE improvement from KF smoothing alone (0%-rejection routes): '
            f'{avg_ne_delta:+.1f} m**',
            '',
            ('Heading limiter is likely needed for mild oscillation routes '
             '(NE_delta < 5m) where gate does not fire.' if avg_ne_delta < 5.0
             else 'KF smoothing alone provides meaningful NE improvement; '
                  'heading limiter is a secondary enhancement.'),
        ]

    report_path = Path('docs/replay_report.md')
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text('\n'.join(report_lines))
    print(f'\nReport written to {report_path}')
    print(f'Plots in {out_dir}/')


if __name__ == '__main__':
    main()

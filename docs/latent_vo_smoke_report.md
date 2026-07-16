# Latent VO Phase 0 Smoke Report

- Mode: `3d` (task spec asks for 3d-first; 2d is phase correlation's strong suit so it is not the target failure mode for latent VO)
- Model: `Bearing_UAV/cross_view`
- Pairs evaluated: 40
- Step size: 64.0 m FOV patches, 25 m nominal nav step

## Correlation-machinery sanity check

Before trusting any real-pair number below, `correlate_volume`/`find_peak_subpixel` are checked against synthetic integer-cell shifts applied to one real backbone feature map (same feature map vs. itself shifted by a known amount -- not a real second frame). This isolates "is the correlation code correct" from "are the encoder's features useful for VO".

| true (dx,dy) cells | estimated (dx,dy) cells | peak | peak ratio |
|---|---|---|---|
| (3, 0) | (3.02, 0.00) | 1.000 | 2.75 |
| (0, -4) | (0.00, -4.01) | 1.000 | 2.75 |
| (5, 5) | (5.02, 5.01) | 1.000 | 2.85 |
| (-6, 2) | (-6.01, 2.01) | 1.000 | 2.94 |

All synthetic shifts recovered to within 0.02 cells with peak ratio >= 2.75 -- the correlation machinery itself is correct. Any localization failure on real pairs below is therefore a property of the encoder's features on genuinely different frames, not a bug.

## Displacement error by method (metres)

| method | mean | std | median | n |
|---|---|---|---|---|
| phase correlation (existing v2 path) | 20.58 | 3.37 | 19.60 | 6 valid / 34 invalid |
| pixel-space direct correlation (no encoder) | 33.35 | 0.50 | 33.28 | 40 valid / 0 invalid |
| latent (backbone tap, 32x32x512) | 28.86 | 16.41 | 27.47 | 40 valid / 0 invalid |
| latent (non-local tap, 16x16x256) | 31.36 | 17.57 | 32.01 | 40 valid / 0 invalid |

## Peak sharpness (peak / 2nd-best-outside-3x3, higher = sharper)

| method | mean | std | median |
|---|---|---|---|
| pixel-space direct | 1.00 | 0.00 | 1.00 |
| latent backbone | 1.08 | 0.14 | 1.03 |
| latent non-local | 1.26 | 0.46 | 1.12 |

Risk (a) check (encoder invariance too strong -> blunt peak): median backbone ratio = 1.03, median non-local ratio = 1.12. Below 1.2 -> invariance-too-strong suspected.

## Tap comparison (backbone vs non-local)

- Backbone tap (32x32x512, cell=2m): mean err 28.86 m, std 16.41 m
- Non-local tap (16x16x256, cell=4m): mean err 31.36 m, std 17.57 m
- Lower std tap: **backbone**

## Rotation smoke (reference only, not a PASS/FAIL gate)

- n=10, mean angle error = 58.72 deg, median = 49.57 deg
| true dtheta (deg) | est dtheta (deg) | err (deg) |
|---|---|---|
| 108.3 | -10.0 | 118.3 |
| -90.8 | -10.0 | 80.8 |
| -77.3 | -6.0 | 71.3 |
| 70.3 | 7.0 | 63.2 |
| 63.4 | 7.4 | 55.9 |
| 41.2 | 5.5 | 35.7 |
| -35.0 | 4.8 | 39.8 |
| -34.4 | 5.4 | 39.8 |
| -33.2 | 10.0 | 43.2 |
| -29.1 | 10.0 | 39.1 |

## Verdict

**FAIL**

Best tap error std = 16.41 m (target < 5 m: NOT MET); median peak ratio = 1.03 (target >= 1.5: NOT MET). See per-pair detail and tap comparison below for root cause before deciding whether to proceed.

## Root cause analysis

The correlation-machinery sanity check above rules out an implementation bug: the exact same `correlate_volume` code recovers synthetic integer shifts on a real feature map with peak ratio >= 2.75. On genuinely different consecutive real frames, however, median peak ratio collapses to ~1.03 (a flat correlation surface -- the argmax is essentially noise) and displacement error is uncorrelated with the true 20-25 m step, often landing near the edge of the search window.

This points to the CSMG/non-local block doing exactly what it was trained to do: `CSMG.forward` L2-normalizes per-cell features, then reconstructs them by a *soft cluster assignment* (`soft_assign` in `cvphr/sceneGraphEncodingNet/nets.py`) whose entire purpose is to make the descriptor robust to exactly the kind of frame-to-frame pixel-position shift this task tries to correlate on. The tap resolution is also very coarse for this purpose: 32x32 cells (2 m/cell) or 16x16 (4 m/cell) over a 64 m FOV, with a 25 m nominal step -- a 12.5-cell displacement is being asked of a representation whose downstream (`descriptor_flatten`) branch discards absolute spatial position entirely (only `JointNet`'s auxiliary `position` field keeps any location info, and only as a soft cluster-weighted centroid, not a dense correlatable map). In short: the encoder was built to answer "which semantic region is this" invariant to viewpoint, not "how many pixels did this region move" -- the two objectives are in tension, and for this tap/resolution the former wins.

Per the task's absolute constraints, this Phase 0 FAIL means Phase 1 (LatentVO module), Phase 2 (fusion integration), and Phase 3 (route evaluation) are **not** started. This report is the complete deliverable for this task.

Possible follow-ups if this direction is revisited later (not undertaken here, listed only for the record): (1) tap an earlier, higher-resolution, less-invariant layer (e.g. before the non-local block, or a shallower VGG stage) at the cost of an extra forward path; (2) supervise a small auxiliary correlation head explicitly for translation (would violate the "no retraining" constraint as given); (3) revisit whether existing phase correlation's 3.4 m std on the 15% of pairs where it does return a result is a more tractable target for improving VO reliability than adding a new latent path.

## Per-pair detail

| run | frame | true dist (m) | phase err | pixel err | latent-bb err | latent-nl err | bb ratio | nl ratio |
|---|---|---|---|---|---|---|---|---|
| nav_34bc50_s25_t20_d3d_phr5_20260705_130157 | 2 | 20.4 | 20.4 | 34.0 | 20.4 | 46.9 | 1.03 | 1.09 |
| nav_34bc51_s25_t20_d3d_phr5_20260705_130328 | 2 | 21.4 | invalid | 33.7 | 29.4 | 24.8 | 1.01 | 1.10 |
| nav_36bc50_s25_t20_d3d_phr5_20260705_130518 | 2 | 22.7 | invalid | 33.3 | 19.2 | 18.2 | 1.11 | 1.20 |
| nav_36bc51_s25_t20_d3d_phr5_20260705_130657 | 2 | 23.2 | invalid | 33.2 | 32.4 | 3.4 | 1.00 | 1.00 |
| nav_37bc50_s25_t20_d3d_phr5_20260705_130829 | 2 | 25.0 | invalid | 32.8 | 19.2 | 22.4 | 1.02 | 1.09 |
| nav_37bc51_s25_t20_d3d_phr5_20260705_131013 | 2 | 25.0 | invalid | 32.8 | 10.0 | 40.1 | 1.00 | 1.00 |
| nav_38bc50_s25_t20_d3d_phr5_20260705_131205 | 2 | 19.9 | invalid | 34.2 | 19.9 | 50.2 | 1.03 | 1.02 |
| nav_38bc51_s25_t20_d3d_phr5_20260705_131317 | 2 | 21.7 | invalid | 33.6 | 0.9 | 2.4 | 1.11 | 1.59 |
| nav_34bc50_s25_t20_d3d_phr5_20260705_130157 | 3 | 20.7 | invalid | 33.9 | 55.0 | 56.4 | 1.04 | 1.00 |
| nav_34bc51_s25_t20_d3d_phr5_20260705_130328 | 3 | 20.5 | invalid | 34.0 | 39.1 | 38.1 | 1.03 | 1.02 |
| nav_36bc50_s25_t20_d3d_phr5_20260705_130518 | 3 | 23.1 | invalid | 33.2 | 59.6 | 58.0 | 1.02 | 1.00 |
| nav_36bc51_s25_t20_d3d_phr5_20260705_130657 | 3 | 23.9 | 18.7 | 33.0 | 19.3 | 48.7 | 1.04 | 1.15 |
| nav_37bc50_s25_t20_d3d_phr5_20260705_130829 | 3 | 25.0 | invalid | 32.8 | 1.6 | 19.7 | 1.04 | 1.10 |
| nav_37bc51_s25_t20_d3d_phr5_20260705_131013 | 3 | 25.0 | invalid | 32.8 | 42.7 | 28.1 | 1.01 | 1.07 |
| nav_38bc50_s25_t20_d3d_phr5_20260705_131205 | 3 | 20.5 | invalid | 34.0 | 11.5 | 11.5 | 1.03 | 1.11 |
| nav_38bc51_s25_t20_d3d_phr5_20260705_131317 | 3 | 20.8 | invalid | 33.9 | 8.9 | 8.6 | 1.04 | 1.12 |
| nav_34bc50_s25_t20_d3d_phr5_20260705_130157 | 4 | 20.8 | invalid | 33.9 | 20.7 | 51.2 | 1.05 | 1.02 |
| nav_34bc51_s25_t20_d3d_phr5_20260705_130328 | 4 | 21.4 | invalid | 33.7 | 34.5 | 34.5 | 1.01 | 1.11 |
| nav_36bc50_s25_t20_d3d_phr5_20260705_130518 | 4 | 22.9 | invalid | 33.3 | 58.6 | 40.6 | 1.05 | 1.23 |
| nav_36bc51_s25_t20_d3d_phr5_20260705_130657 | 4 | 23.2 | invalid | 33.2 | 37.7 | 38.9 | 1.03 | 1.04 |
| nav_37bc50_s25_t20_d3d_phr5_20260705_130829 | 4 | 25.0 | invalid | 32.8 | 39.4 | 13.6 | 1.01 | 1.22 |
| nav_37bc51_s25_t20_d3d_phr5_20260705_131013 | 4 | 25.0 | 24.5 | 32.8 | 22.0 | 18.6 | 1.03 | 1.08 |
| nav_38bc50_s25_t20_d3d_phr5_20260705_131205 | 4 | 20.0 | invalid | 34.2 | 32.0 | 32.0 | 1.08 | 1.22 |
| nav_38bc51_s25_t20_d3d_phr5_20260705_131317 | 4 | 22.6 | invalid | 33.4 | 45.9 | 9.9 | 1.01 | 1.25 |
| nav_34bc50_s25_t20_d3d_phr5_20260705_130157 | 5 | 21.2 | invalid | 33.8 | 54.6 | 41.2 | 1.03 | 1.08 |
| nav_34bc51_s25_t20_d3d_phr5_20260705_130328 | 5 | 24.3 | invalid | 32.9 | 32.0 | 32.0 | 1.13 | 1.48 |
| nav_36bc50_s25_t20_d3d_phr5_20260705_130518 | 5 | 22.8 | invalid | 33.3 | 21.0 | 20.9 | 1.04 | 1.18 |
| nav_36bc51_s25_t20_d3d_phr5_20260705_130657 | 5 | 24.9 | invalid | 32.8 | 24.1 | 11.4 | 1.03 | 1.03 |
| nav_37bc50_s25_t20_d3d_phr5_20260705_130829 | 5 | 25.0 | invalid | 32.8 | 33.2 | 62.2 | 1.01 | 1.06 |
| nav_37bc51_s25_t20_d3d_phr5_20260705_131013 | 5 | 25.0 | invalid | 32.8 | 42.2 | 41.6 | 1.09 | 1.45 |
| nav_38bc50_s25_t20_d3d_phr5_20260705_131205 | 5 | 20.2 | 25.4 | 34.1 | 61.2 | 56.6 | 1.03 | 1.20 |
| nav_38bc51_s25_t20_d3d_phr5_20260705_131317 | 5 | 23.5 | 15.8 | 33.1 | 12.0 | 11.8 | 1.14 | 1.21 |
| nav_34bc50_s25_t20_d3d_phr5_20260705_130157 | 6 | 21.1 | 18.8 | 33.8 | 20.7 | 33.8 | 1.03 | 1.01 |
| nav_34bc51_s25_t20_d3d_phr5_20260705_130328 | 6 | 22.5 | invalid | 33.4 | 32.0 | 32.0 | 1.08 | 1.21 |
| nav_36bc50_s25_t20_d3d_phr5_20260705_130518 | 6 | 23.7 | invalid | 33.0 | 11.6 | 64.3 | 1.01 | 1.44 |
| nav_36bc51_s25_t20_d3d_phr5_20260705_130657 | 6 | 25.0 | invalid | 32.8 | 12.2 | 12.0 | 1.63 | 3.47 |
| nav_37bc50_s25_t20_d3d_phr5_20260705_130829 | 6 | 25.0 | invalid | 32.8 | 33.2 | 33.4 | 1.33 | 1.70 |
| nav_37bc51_s25_t20_d3d_phr5_20260705_131013 | 6 | 25.0 | invalid | 32.8 | 25.5 | 25.5 | 1.05 | 1.17 |
| nav_38bc50_s25_t20_d3d_phr5_20260705_131205 | 6 | 21.2 | invalid | 33.8 | 55.3 | 55.3 | 1.05 | 1.36 |
| nav_38bc51_s25_t20_d3d_phr5_20260705_131317 | 6 | 21.0 | invalid | 33.8 | 3.6 | 3.7 | 1.66 | 2.69 |
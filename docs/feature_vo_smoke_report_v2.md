# Feature VO Phase 0 Smoke Report -- v2 (theta-free)

- Mode: `3d` · pairs: 40 (same harness/pairs as v1 and the latent report)
- v1 (`docs/feature_vo_smoke_report.md`) judged ORB+RANSAC against a world-frame ground truth composed using the nav log's `theta`. This session found that `theta` is not trustworthy for 3d mode's `_find_nearest_v3d`-sourced frames (no heading field in the v3d library at all), and that warping frame t-1 by the estimated homography reproduces frame t almost exactly on most pairs -- direct evidence the estimator itself is largely correct. v2 drops `theta` from validation entirely.

## What's validated here, and what isn't

| | v1 | v2 |
|---|---|---|
| Validity gate | n_matches, inlier_ratio | + reprojection MAE (does warp(frame t-1, H) actually match frame t?) |
| Accuracy check | world-frame (east, north) vs. theta-composed ground truth | displacement **magnitude** vs. true step distance (from real lon/lat, theta-independent) |
| Direction / heading | compared to logged theta (rejected as unreliable) | logged for reference only, not validated |

## Valid rate and rejection reasons

- valid rate: **70%** (28/40)
| reason | count |
|---|---|
| ok | 28 |
| poor_reprojection | 7 |
| few_matches | 5 |

## Displacement magnitude error (valid pairs, metres, theta-independent)

| | mean (signed bias) | std | mean abs | median abs | n |
|---|---|---|---|---|---|
| magnitude error | -0.80 | 6.66 | 5.30 | 3.81 | 28 |

Reprojection MAE on valid pairs: mean 11.53, median 12.28 (0-255 scale; for reference, raw unwarped-pair MAE runs 40-70 in this dataset, so anything well below that indicates real alignment, not chance).

## Verdict

**PARTIAL**

Valid rate 70%, magnitude error std 6.66 m, |mean bias| 0.80 m -- borderline; see per-pair detail before deciding.

## Known gap -- direction is still unvalidated

This report does not establish that the estimated **direction** (heading-composed world-frame vector) is correct -- only that the magnitude is, and that the homography self-consistently reprojects. Composing a displacement into world (lon, lat) still requires *some* heading estimate, and the only two candidates are (a) the logged `theta` (shown unreliable) or (b) ORB/homography's own rotation estimate, e.g. `est_yaw_deg` logged per-pair below (untested against any ground truth). Before this can feed the fusion filter, direction needs an independent validation path -- e.g. chaining several steps and checking the resulting trajectory shape against the known waypoint path, or finding/recovering true per-image capture heading if it exists anywhere in the v3d generation pipeline.

## Per-pair detail

| run | frame | true (m) | pred (m) | mag err (m) | warp MAE | inlier% | n_good | est_yaw | logged Δθ | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| 34bc50 | 2 | 20.4 | 16.5 | -4.0 | 7.4 | 99 | 520 | +45.7 | +10.5 | ok |
| 34bc51 | 2 | 21.4 | 11.4 | -10.0 | 11.0 | 96 | 718 | +19.7 | -6.9 | ok |
| 36bc50 | 2 | 22.7 | 18.2 | -4.6 | 5.8 | 98 | 779 | -104.0 | +16.3 | ok |
| 36bc51 | 2 | 23.2 | 7.3 | -15.9 | 9.6 | 96 | 950 | +34.7 | -8.5 | ok |
| 37bc50 | 2 | 25.0 | 28.7 | +3.7 | 14.0 | 77 | 93 | -160.6 | +2.0 | ok |
| 37bc51 | 2 | 25.0 | - | - | - | - | 21 | - | -13.5 | few_matches |
| 38bc50 | 2 | 19.9 | - | - | - | - | 26 | - | +10.5 | few_matches |
| 38bc51 | 2 | 21.7 | 24.6 | +2.9 | 14.8 | 97 | 388 | -16.8 | -5.6 | ok |
| 34bc50 | 3 | 20.7 | 24.2 | +3.5 | 13.4 | 92 | 157 | -31.2 | -8.8 | ok |
| 34bc51 | 3 | 20.5 | 20.2 | -0.2 | 17.0 | 76 | 428 | -128.7 | +21.0 | ok |
| 36bc50 | 3 | 23.1 | 13.4 | -9.7 | 7.5 | 100 | 984 | +159.9 | -15.1 | ok |
| 36bc51 | 3 | 23.9 | - | - | - | - | 16 | - | +19.3 | few_matches |
| 37bc50 | 3 | 25.0 | 17.7 | -7.3 | 8.8 | 97 | 404 | +42.8 | -9.4 | ok |
| 37bc51 | 3 | 25.0 | - | - | 20.8 | 90 | 226 | - | +8.2 | poor_reprojection |
| 38bc50 | 3 | 20.5 | - | - | - | - | 6 | - | -14.4 | few_matches |
| 38bc51 | 3 | 20.8 | 17.3 | -3.5 | 12.1 | 98 | 435 | +102.7 | +11.8 | ok |
| 34bc50 | 4 | 20.8 | 20.1 | -0.7 | 13.0 | 76 | 244 | -79.1 | -2.5 | ok |
| 34bc51 | 4 | 21.4 | - | - | - | - | 15 | - | -21.4 | few_matches |
| 36bc50 | 4 | 22.9 | 30.0 | +7.1 | 8.7 | 88 | 382 | +95.8 | +6.6 | ok |
| 36bc51 | 4 | 23.2 | - | - | 24.5 | 77 | 304 | - | -18.7 | poor_reprojection |
| 37bc50 | 4 | 25.0 | 30.4 | +5.4 | 12.5 | 87 | 207 | -84.6 | -8.5 | ok |
| 37bc51 | 4 | 25.0 | 19.1 | -5.9 | 9.7 | 90 | 540 | -23.1 | -33.2 | ok |
| 38bc50 | 4 | 20.0 | 35.6 | +15.6 | 3.4 | 98 | 81 | -115.8 | +41.2 | ok |
| 38bc51 | 4 | 22.6 | 24.9 | +2.3 | 16.0 | 82 | 253 | -78.0 | -21.5 | ok |
| 34bc50 | 5 | 21.2 | 19.6 | -1.6 | 16.9 | 90 | 277 | +116.1 | -7.0 | ok |
| 34bc51 | 5 | 24.3 | 31.8 | +7.6 | 16.3 | 92 | 137 | +152.6 | -77.3 | ok |
| 36bc50 | 5 | 22.8 | 24.3 | +1.5 | 9.3 | 97 | 147 | +148.0 | +4.5 | ok |
| 36bc51 | 5 | 24.9 | 19.5 | -5.4 | 16.9 | 98 | 551 | -80.6 | +70.3 | ok |
| 37bc50 | 5 | 25.0 | - | - | 23.8 | 65 | 288 | - | -7.1 | poor_reprojection |
| 37bc51 | 5 | 25.0 | 32.7 | +7.7 | 7.3 | 90 | 48 | -113.3 | -90.8 | ok |
| 38bc50 | 5 | 20.2 | 23.1 | +2.9 | 5.1 | 92 | 346 | -148.4 | -35.0 | ok |
| 38bc51 | 5 | 23.5 | 25.8 | +2.3 | 15.4 | 92 | 202 | +17.1 | -10.0 | ok |
| 34bc50 | 6 | 21.1 | - | - | 19.9 | 83 | 41 | - | +0.8 | poor_reprojection |
| 34bc51 | 6 | 22.5 | 23.0 | +0.6 | 12.5 | 94 | 389 | -152.3 | +63.4 | ok |
| 36bc50 | 6 | 23.7 | - | - | 22.8 | 73 | 254 | - | -29.1 | poor_reprojection |
| 36bc51 | 6 | 25.0 | - | - | 30.4 | 83 | 138 | - | -15.2 | poor_reprojection |
| 37bc50 | 6 | 25.0 | 21.6 | -3.4 | 19.4 | 77 | 144 | -6.7 | -34.4 | ok |
| 37bc51 | 6 | 25.0 | 13.4 | -11.6 | 5.9 | 91 | 101 | +32.7 | +108.3 | ok |
| 38bc50 | 6 | 21.2 | - | - | 16.1 | 60 | 43 | - | -16.9 | poor_reprojection |
| 38bc51 | 6 | 21.0 | 19.3 | -1.7 | 13.2 | 95 | 480 | +1.1 | +28.9 | ok |
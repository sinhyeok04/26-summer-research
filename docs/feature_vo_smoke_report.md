# Feature VO (ORB+RANSAC) Phase 0 Smoke Report

- Mode: `3d`
- Pairs evaluated: 40 (same harness/pairs as `docs/latent_vo_smoke_report.md`)
- rotation sign convention used: dtheta_est_deg = -1 * atan2(c,a) of the estimated similarity matrix (image y-down vs. world CCS y-up mirrors rotation sense)

## Coordinate-conversion sanity check (found and fixed during this task)

`ego_to_world_m` (shared with `scripts/smoke_latent_vo.py`) originally assumed image "up" = UAV forward. Debugging an initial ~23 m systematic bias on clean 2d control pairs (std only ~2.6 m -- too tight to be noise) showed predicted vectors were consistently the true vector rotated by exactly -90 deg. Root cause: for `crop_target_patch` (2d), forward maps onto the image's **x** axis (right), not "up" -- at theta=0 (heading=east) no rotation is applied, so the plain north-up crop has east/forward pointing image-right. Fixed to `forward_m=-dx, right_m=-dy`; re-verified on the same control pairs (below) to ~0.2 m mean error. This fix was applied to both this script and the latent-VO smoke script; re-running the latter confirmed its FAIL verdict is unchanged (root cause there was a flat correlation surface, independent of this coordinate bug).

## 2d control check (same algorithm, deterministically heading-aligned data)

Same `estimate_orb_ransac` + `ego_to_world_m` code, run on `2d` `crop_target_patch` pairs instead of `3d` v3d-library pairs, to isolate "is the algorithm/code correct" from "is 3d's frame-to-frame data actually describable by a simple 2D rigid transform".

- valid rate: 100% (n_total=40)
- error (valid pairs): mean 0.18 m, std 0.09 m, median 0.19 m (n=40)

## Magnitude-only check (3d, direction ignored)

For valid 3d pairs, does `|(dx_px,dy_px)| * m_per_px` (ego-frame displacement magnitude, ignoring the heading composition / world direction entirely) correlate with the true step distance? If even magnitude is uncorrelated, the failure is not just "wrong heading used to compose direction" -- the raw pixel correspondence itself is not describing the two frames' true relative motion.

- correlation(pred magnitude, true distance) = -0.11 (n=28), mean predicted magnitude 21.1 m vs. mean true 22.7 m

## Direction sanity check (3d)

Mean cosine similarity between predicted and true displacement vectors (valid pairs, plain): 0.22 (median 0.49) on 3d data, vs. the 2d control check above where the same code is near-perfectly accurate. Values near +1 would mean the estimate is directionally correct; near -1 a residual sign flip; near 0 (as found) directionally uninformative -- consistent with the magnitude-only check also finding no correlation, not merely a coordinate-convention artifact.

## Valid rate and rejection-reason breakdown

| variant | valid rate | low_inlier_ratio | ok |
|---|---|---|---|
| plain | 70% | 12 | 28 |
| CLAHE | 60% | 16 | 24 |

## Displacement error, valid pairs only (metres)

| method | mean | std | median | n valid |
|---|---|---|---|---|
| phase correlation (existing v2 path) | 20.58 | 3.37 | 19.60 | 6 / 40 |
| ORB+RANSAC (plain) | 24.19 | 13.17 | 22.03 | 28 / 40 |
| ORB+RANSAC (CLAHE) | 22.32 | 12.69 | 21.49 | 24 / 40 |

Cited for reference (from `docs/latent_vo_smoke_report.md`, not recomputed here): latent backbone tap mean 33.2 m / std 14.4 m, latent non-local tap mean 33.4 m / std 14.3 m, both over the same 40 pairs -- latent VO was rejected at Phase 0.

## Systematic bias check (signed, along true-direction component)

`bias_m` = component of (predicted - true) displacement vector projected onto the true direction; positive = overshoot, negative = undershoot. This is the metric the PASS gate's `|mean error| < 3 m` checks -- distinct from the (always >= 0) Euclidean `err_m` used for std.

- plain: mean bias = -17.65 m, std = 16.26 m (n=28)

## Rotation error (reference, ORB+RANSAC, plain; not a PASS/FAIL gate)

- mean = 75.98 deg, median = 84.99 deg (n=28)
- median >= 3 deg -> not yet precise enough as a standalone heading source.

## Verdict

**FAIL**

Valid rate 70% (< 30%) or error std 13.17 m (>= 10 m). Do not proceed to Phase 1; see rejection-reason breakdown for root cause.

## Root cause analysis

The 2d control check rules out a code/algorithm bug: the exact same `estimate_orb_ransac` + `ego_to_world_m` pipeline is close to sub-metre accurate with 100% valid rate on 2d `crop_target_patch` pairs. On 3d pairs it is not just directionally wrong (cos_sim ~0, not ~-1) -- the magnitude-only check shows the ego-frame displacement magnitude *itself* barely correlates with the true step distance. That rules out "just a wrong heading was used to compose direction" as the (sole) explanation: the raw point correspondences RANSAC is voting on do not describe the true rigid motion between the two images at all, regardless of what direction they're projected in.

Structural cause, traced in `naver/runners/nav.py`'s `get_patches` / `_find_nearest_v3d`: 3d mode's target patch is not a continuously-rendered flight frame -- it is the **nearest pre-captured library image by (lat, lng) only** (`_find_nearest_v3d` never looks at heading). The library JSON metadata (e.g. `Bearing_UAV_90K/citya/uav_254k_34bc_b15_s100/blk_6_12_s_081_v3d.json`) has no heading/yaw/roll field at all, and its `left_mid`/`right_mid`/`top_mid`/`bottom_mid` bounding box is axis-aligned (north-up), consistent with each of the ~200 samples per block being an independently-generated cross-view render rather than a frame from one continuous, heading-consistent trajectory. Two nav steps 25 m apart can therefore land on two library samples whose *actual* rendered viewpoints differ arbitrarily -- exactly the "parallax / rotation / occlusion" 3d conditions the original latent-VO task called out, but here it means the fundamental assumption every frame-to-frame VO method in this task family relies on ("two temporally-close frames differ by close to a simple 2D rigid transform") does not hold for 3d mode's data source at all -- independent of whether the VO method itself is phase correlation, latent correlation, or ORB+RANSAC.

Per the task's absolute constraints, this Phase 0 FAIL means Phase 1 (FeatureVO module), Phase 2 (3d route evaluation) are **not** started. This report, together with the coordinate-conversion fix (now corrected in both `scripts/smoke_latent_vo.py` and this script) and the diagnostics above, is the complete deliverable for this task.

This also reframes the earlier latent-VO FAIL: that report's root cause (encoder invariance destroying position information) is still independently supported by its own synthetic-shift sanity check, but the v3d-library viewpoint-diversity issue found here would have capped *any* 3d frame-to-frame VO approach regardless. Follow-up worth flagging for whoever owns the 3d data pipeline (not undertaken here): making `_find_nearest_v3d` heading-aware, or sourcing a genuinely continuous 3d flight-video dataset, would be a precondition for any frame-to-frame VO working in 3d mode -- not something fixable from the VO algorithm side alone.

## Per-pair detail

| run | frame | true (m) | phase err | plain valid | plain reason | n_match | inlier% | plain err | plain rot err |
|---|---|---|---|---|---|---|---|---|---|
| nav_34bc50_s25_t20_d3d_phr5_20260705_130157 | 2 | 20.4 | 20.4 | True | ok | 494 | 73 | 27.3 | 56.6 |
| nav_34bc51_s25_t20_d3d_phr5_20260705_130328 | 2 | 21.4 | invalid | True | ok | 558 | 80 | 29.9 | 13.4 |
| nav_36bc50_s25_t20_d3d_phr5_20260705_130518 | 2 | 22.7 | invalid | True | ok | 569 | 74 | 19.2 | 87.7 |
| nav_36bc51_s25_t20_d3d_phr5_20260705_130657 | 2 | 23.2 | invalid | True | ok | 661 | 92 | 16.3 | 26.3 |
| nav_37bc50_s25_t20_d3d_phr5_20260705_130829 | 2 | 25.0 | invalid | False | low_inlier_ratio | 308 | 25 | - | - |
| nav_37bc51_s25_t20_d3d_phr5_20260705_131013 | 2 | 25.0 | invalid | False | low_inlier_ratio | 257 | 5 | - | - |
| nav_38bc50_s25_t20_d3d_phr5_20260705_131205 | 2 | 19.9 | invalid | True | ok | 79 | 37 | 45.8 | 125.6 |
| nav_38bc51_s25_t20_d3d_phr5_20260705_131317 | 2 | 21.7 | invalid | True | ok | 439 | 53 | 8.9 | 24.5 |
| nav_34bc50_s25_t20_d3d_phr5_20260705_130157 | 3 | 20.7 | invalid | False | low_inlier_ratio | 278 | 22 | - | - |
| nav_34bc51_s25_t20_d3d_phr5_20260705_130328 | 3 | 20.5 | invalid | True | ok | 447 | 57 | 18.8 | 107.9 |
| nav_36bc50_s25_t20_d3d_phr5_20260705_130518 | 3 | 23.1 | invalid | True | ok | 635 | 85 | 22.0 | 144.7 |
| nav_36bc51_s25_t20_d3d_phr5_20260705_130657 | 3 | 23.9 | 18.7 | False | low_inlier_ratio | 312 | 3 | - | - |
| nav_37bc50_s25_t20_d3d_phr5_20260705_130829 | 3 | 25.0 | invalid | True | ok | 462 | 70 | 22.0 | 33.8 |
| nav_37bc51_s25_t20_d3d_phr5_20260705_131013 | 3 | 25.0 | invalid | True | ok | 353 | 46 | 3.4 | 47.3 |
| nav_38bc50_s25_t20_d3d_phr5_20260705_131205 | 3 | 20.5 | invalid | False | low_inlier_ratio | 117 | 4 | - | - |
| nav_38bc51_s25_t20_d3d_phr5_20260705_131317 | 3 | 20.8 | invalid | True | ok | 445 | 62 | 31.6 | 114.8 |
| nav_34bc50_s25_t20_d3d_phr5_20260705_130157 | 4 | 20.8 | invalid | True | ok | 365 | 42 | 37.5 | 82.2 |
| nav_34bc51_s25_t20_d3d_phr5_20260705_130328 | 4 | 21.4 | invalid | False | low_inlier_ratio | 265 | 3 | - | - |
| nav_36bc50_s25_t20_d3d_phr5_20260705_130518 | 4 | 22.9 | invalid | True | ok | 432 | 69 | 51.6 | 102.1 |
| nav_36bc51_s25_t20_d3d_phr5_20260705_130657 | 4 | 23.2 | invalid | True | ok | 410 | 38 | 41.2 | 3.2 |
| nav_37bc50_s25_t20_d3d_phr5_20260705_130829 | 4 | 25.0 | invalid | True | ok | 384 | 52 | 13.7 | 92.8 |
| nav_37bc51_s25_t20_d3d_phr5_20260705_131013 | 4 | 25.0 | 24.5 | True | ok | 497 | 70 | 23.4 | 57.3 |
| nav_38bc50_s25_t20_d3d_phr5_20260705_131205 | 4 | 20.0 | invalid | True | ok | 160 | 40 | 21.4 | 74.3 |
| nav_38bc51_s25_t20_d3d_phr5_20260705_131317 | 4 | 22.6 | invalid | True | ok | 381 | 55 | 8.7 | 99.1 |
| nav_34bc50_s25_t20_d3d_phr5_20260705_130157 | 5 | 21.2 | invalid | True | ok | 390 | 31 | 34.1 | 92.8 |
| nav_34bc51_s25_t20_d3d_phr5_20260705_130328 | 5 | 24.3 | invalid | True | ok | 275 | 31 | 9.7 | 74.3 |
| nav_36bc50_s25_t20_d3d_phr5_20260705_130518 | 5 | 22.8 | invalid | False | low_inlier_ratio | 312 | 28 | - | - |
| nav_36bc51_s25_t20_d3d_phr5_20260705_130657 | 5 | 24.9 | invalid | True | ok | 512 | 70 | 6.8 | 10.1 |
| nav_37bc50_s25_t20_d3d_phr5_20260705_130829 | 5 | 25.0 | invalid | True | ok | 418 | 45 | 30.4 | 110.6 |
| nav_37bc51_s25_t20_d3d_phr5_20260705_131013 | 5 | 25.0 | invalid | False | low_inlier_ratio | 146 | 20 | - | - |
| nav_38bc50_s25_t20_d3d_phr5_20260705_131205 | 5 | 20.2 | 25.4 | True | ok | 339 | 80 | 27.2 | 176.7 |
| nav_38bc51_s25_t20_d3d_phr5_20260705_131317 | 5 | 23.5 | 15.8 | True | ok | 347 | 45 | 15.2 | 6.5 |
| nav_34bc50_s25_t20_d3d_phr5_20260705_130157 | 6 | 21.1 | 18.8 | False | low_inlier_ratio | 327 | 3 | - | - |
| nav_34bc51_s25_t20_d3d_phr5_20260705_130328 | 6 | 22.5 | invalid | True | ok | 422 | 76 | 41.8 | 89.4 |
| nav_36bc50_s25_t20_d3d_phr5_20260705_130518 | 6 | 23.7 | invalid | True | ok | 330 | 36 | 45.0 | 102.8 |
| nav_36bc51_s25_t20_d3d_phr5_20260705_130657 | 6 | 25.0 | invalid | False | low_inlier_ratio | 347 | 19 | - | - |
| nav_37bc50_s25_t20_d3d_phr5_20260705_130829 | 6 | 25.0 | invalid | False | low_inlier_ratio | 314 | 23 | - | - |
| nav_37bc51_s25_t20_d3d_phr5_20260705_131013 | 6 | 25.0 | invalid | True | ok | 162 | 75 | 21.0 | 140.2 |
| nav_38bc50_s25_t20_d3d_phr5_20260705_131205 | 6 | 21.2 | invalid | False | low_inlier_ratio | 293 | 4 | - | - |
| nav_38bc51_s25_t20_d3d_phr5_20260705_131317 | 6 | 21.0 | invalid | True | ok | 453 | 62 | 3.3 | 30.4 |
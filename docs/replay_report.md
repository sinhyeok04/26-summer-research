# Phase 1: KF Gate Replay Report

Q=100.0 m², R=600.0 m², gate=5.99

## Per-Route Summary

| route | fail? | steps | maxWP | reject% | d²_fail | d²_ok | ratio_fail | NE_kf_m | NE_model_m | NE_delta_m |
|-------|-------|-------|-------|---------|---------|-------|------------|---------|------------|------------|
| 34bc_50 | FAIL | 24 | 5 | 4.2% | 0.77 | 0.56 | 12.5% | 13.2 | 28.7 | +15.5 |
| 34bc_51 | OK | 51 | 8 | 0.0% | 1.41 | 0.63 | 7.8% | 14.9 | 23.5 | +8.6 |
| 36bc_50 | FAIL | 44 | 11 | 0.0% | 1.63 | 0.35 | 9.1% | 9.9 | 19.2 | +9.3 |
| 36bc_51 | OK | 43 | 9 | 0.0% | 2.07 | 0.31 | 2.3% | 10.5 | 19.9 | +9.5 |
| 37bc_50 | OK | 45 | 11 | 0.0% | 0.89 | 0.71 | 4.4% | 8.1 | 22.1 | +14.0 |
| 37bc_51 | OK | 61 | 9 | 0.0% | 1.10 | 0.56 | 4.9% | 11.1 | 22.7 | +11.7 |
| 38bc_50 | FAIL | 52 | 9 | 7.7% | 4.24 | 0.66 | 19.2% | 54.6 | 68.9 | +14.3 |
| 38bc_51 | FAIL | 70 | 6 | 34.3% | 20.86 | 0.84 | 34.3% | 127.2 | 236.8 | +109.5 |

## Verdict

**Hypothesis**: In failing routes, d² at failing segments should be significantly higher than in succeeding segments.

- Avg d² in **failing** route failing segments: **6.88**
- Avg d² in **succeeding** route ok segments:   **0.55**
- Avg reject rate in failing routes:  11.5%
- Avg reject rate in succeeding routes: 0.0%

**PASS** — d² is 12.4× higher in failing segments, and reject rate in succeeding routes is 0.0% < 20%. Proceed to Phase 2.

## Phase 1(C): KF Smoothing Effect on 0%-Rejection Routes

For routes where the gate never fires, the KF acts as a pure smoother (VO-weighted position update at every step). NE_delta = NE_model − NE_kf: positive means KF smoothing improves navigation accuracy.

| route | reject% | NE_model_m | NE_kf_m | NE_delta_m | interpretation |
|-------|---------|------------|---------|------------|----------------|
| 34bc_51 | 0.0% | 23.5 | 14.9 | +8.6 | smoothing helps |
| 36bc_50 | 0.0% | 19.2 | 9.9 | +9.3 | smoothing helps |
| 36bc_51 | 0.0% | 19.9 | 10.5 | +9.5 | smoothing helps |
| 37bc_50 | 0.0% | 22.1 | 8.1 | +14.0 | smoothing helps |
| 37bc_51 | 0.0% | 22.7 | 11.1 | +11.7 | smoothing helps |

**Avg NE improvement from KF smoothing alone (0%-rejection routes): +10.6 m**

KF smoothing alone provides meaningful NE improvement; heading limiter is a secondary enhancement.
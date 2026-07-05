"""
Unit tests for InnovationGatedFusion (DR-Bearing v2).

Scenarios:
  (a) Model accurate       → reject rate ~0, fused converges toward model_pred
  (b) Model jumps 100m at steps 10-20 → those steps rejected, fused stays on VO path
  (c) Model scattered around wrong location → M-of-N blocks re-anchor (no consensus)
  (d) Soft gate: 40m middle jumps pass hard gate; soft gate on reduces pull vs off

Note on KF gate calibration (Q=100, R=600):
  Steady-state P_ss ≈ 200 m², P_pred ≈ 300 m².
  Rejection threshold: d² > 5.99 → jump² > 5.99*(P_pred+R) = 5.99*900 = 5391 → jump > 73.4m.
  Tests use 100m jump so d²=10000/900≈11.1 >> 5.99.
  Tests warm up 30 accurate steps first to ensure P has converged to steady state.
"""

import math
import unittest
import numpy as np

from naver.fusion.innovation_gated import InnovationGatedFusion, _haversine_m


# ---------------------------------------------------------------------------
# Minimal stub for FrameToFrameVO (avoids cv2 dependency in unit tests)
# ---------------------------------------------------------------------------

class _StubVO:
    """VO stub: returns a fixed (dlon, dlat) displacement each step."""

    def __init__(self, dlon_per_step: float = 0.0001, dlat_per_step: float = 0.0001):
        self.dlon = dlon_per_step
        self.dlat = dlat_per_step
        self.scale_factor = 1.0
        self._ready = False

    def estimate(self, img, theta_deg, scale_m_per_px):
        if not self._ready:
            return None
        return (self.dlon, self.dlat)

    def push(self, img, theta_deg):
        self._ready = True

    def reset(self):
        self._ready = False


def _make_igf(vo, **kw) -> InnovationGatedFusion:
    defaults = dict(
        kf_q=100.0,
        kf_r=600.0,
        chi2_gate=5.99,
        use_soft_gate=False,
        soft_knee=1.0,
        reanchor_after=5,
        mn_m=3,
        mn_n=5,
        consensus_radius=15.0,
        vo_scale_init=1.0,
    )
    defaults.update(kw)
    igf = InnovationGatedFusion(vo=vo, **defaults)
    return igf


def _step(igf, model_lon, model_lat):
    """Single step with a dummy image (StubVO ignores it)."""
    fake_img = np.zeros((256, 256, 3), dtype=np.uint8)
    return igf.step(
        model_pred=(model_lon, model_lat),
        uvp_img=fake_img,
        theta_deg=90.0,
        scale_m_per_px=0.25,
    )


def _warm_up(igf, n=30, base_lon=139.69, base_lat=35.67):
    """Run n accurate steps to converge P to steady state (~200 m²)."""
    lon, lat = base_lon, base_lat
    for _ in range(n):
        lon += 0.0001
        lat += 0.0001
        _step(igf, lon, lat)
    return lon, lat


COS_LAT = math.cos(math.radians(35.67))


def _m_to_dlon(metres: float) -> float:
    return metres / (111320.0 * COS_LAT)


# ---------------------------------------------------------------------------
# (a) Accurate model: gate should always accept
# ---------------------------------------------------------------------------

class TestScenarioA(unittest.TestCase):

    def test_low_reject_rate(self):
        """When model agrees with VO, gate accepts every step."""
        vo = _StubVO(dlon_per_step=0.0001, dlat_per_step=0.0001)
        igf = _make_igf(vo)
        igf.reset()

        base_lon, base_lat = 139.69, 35.67
        n_accept = 0
        lon, lat = base_lon, base_lat
        for i in range(30):
            lon += 0.0001
            lat += 0.0001
            fused, info = _step(igf, lon, lat)
            if i > 0:  # skip step-0 initialisation
                n_accept += info['accept']

        reject_rate = 1.0 - n_accept / 29
        self.assertLess(reject_rate, 0.10,
                        f"Expected <10% reject on accurate model, got {reject_rate:.1%}")

    def test_fused_tracks_model(self):
        """Fused position should stay close to model_pred when model is accurate."""
        vo = _StubVO(dlon_per_step=0.0001, dlat_per_step=0.0001)
        igf = _make_igf(vo)
        igf.reset()

        lon, lat = 139.69, 35.67
        for _ in range(20):
            lon += 0.0001
            lat += 0.0001
            fused, info = _step(igf, lon, lat)

        dist_m = _haversine_m(fused[0], fused[1], lon, lat)
        self.assertLess(dist_m, 30.0,
                        f"Fused should be within 30m of accurate model_pred, got {dist_m:.1f}m")


# ---------------------------------------------------------------------------
# (b) Model jumps 100m at steps 10-20 after warm-up
#
# At steady state P_ss≈200, P_pred≈300, R=600:
#   d²(100m) = 10000/900 = 11.1 >> gate=5.99 → rejected
# ---------------------------------------------------------------------------

class TestScenarioB(unittest.TestCase):

    def _run(self):
        vo = _StubVO(dlon_per_step=0.0001, dlat_per_step=0.0001)
        igf = _make_igf(vo, reanchor_after=5, mn_m=3, mn_n=5, consensus_radius=15.0)
        igf.reset()
        jump_lon = _m_to_dlon(100.0)

        lon, lat = _warm_up(igf)  # bring P to steady state

        results = []
        for i in range(35):
            lon += 0.0001
            lat += 0.0001
            m_lon = lon + jump_lon if 10 <= i < 21 else lon
            fused, info = _step(igf, m_lon, lat)
            results.append((fused, info))
        return results, jump_lon, lon - 35 * 0.0001, lat - 35 * 0.0001  # base_lon/lat after warmup

    def test_jump_steps_rejected(self):
        results, *_ = self._run()
        jump_accepts = [results[i][1]['accept'] for i in range(10, 21)]
        reject_count = jump_accepts.count(0)
        self.assertGreaterEqual(reject_count, 7,
                                f"At least 7/11 jump steps should be rejected, got {reject_count}")

    def test_fused_stays_on_vo_during_jump(self):
        results, jump_lon, base_lon, base_lat = self._run()
        # fused at step 15 (mid-jump) should be near VO trajectory, not the jumped model_pred
        fused_mid, _ = results[15]
        # VO-propagated position at step 16 (1-indexed) after warm-up
        vo_lon_mid = base_lon + 16 * 0.0001
        vo_lat_mid = base_lat + 16 * 0.0001
        jumped_model_lon = vo_lon_mid + jump_lon

        dist_to_jumped = _haversine_m(fused_mid[0], fused_mid[1], jumped_model_lon, vo_lat_mid)
        dist_to_vo = _haversine_m(fused_mid[0], fused_mid[1], vo_lon_mid, vo_lat_mid)
        self.assertLess(dist_to_vo, dist_to_jumped,
                        "Fused should be closer to VO trajectory than jumped model during rejection")


# ---------------------------------------------------------------------------
# (c) Scattered model: M-of-N blocks re-anchor when measurements lack consensus
#
# Approach:
#   1. Warm up to steady state.
#   2. Send 200m-offset measurements for >reanchor_after steps → consec_reject>=5.
#      d²(200m, P_pred≈300) = 40000/900 = 44.4 >> 5.99 → always rejected initially.
#   3. Send alternating ±40m offsets (pass gate due to large P after many rejects).
#      Adjacent measurements are 80m apart > consensus_radius=15m → M-of-N fails.
# ---------------------------------------------------------------------------

class TestScenarioC(unittest.TestCase):

    def _setup_in_reanchor_mode(self, igf, base_lon, base_lat, n_reject=8):
        """Drive igf into re-anchor mode via large-offset rejections."""
        lon, lat = base_lon, base_lat
        offset_lon = _m_to_dlon(200.0)
        for _ in range(n_reject):
            lon += 0.0001
            lat += 0.0001
            _step(igf, lon + offset_lon, lat)
        return lon, lat

    def test_initial_sustained_rejection(self):
        """Large-offset model is rejected in the first ~8 steps (before P grows too large)."""
        vo = _StubVO(dlon_per_step=0.0001, dlat_per_step=0.0001)
        igf = _make_igf(vo, kf_q=100.0, kf_r=600.0, chi2_gate=5.99,
                        reanchor_after=5, mn_m=3, mn_n=5, consensus_radius=15.0)
        igf.reset()
        offset_lon = _m_to_dlon(200.0)

        lon, lat = _warm_up(igf)  # P≈200 at start of test
        accepts = []
        for _ in range(8):
            lon += 0.0001
            lat += 0.0001
            fused, info = _step(igf, lon + offset_lon, lat)
            accepts.append(info['accept'])

        reject_count = accepts.count(0)
        self.assertGreaterEqual(reject_count, 6,
                                f"200m offset should be rejected in most of first 8 steps, got {reject_count}/8")

    def test_mn_blocks_reanchor_on_scattered_model(self):
        """M-of-N blocks re-anchor when gate-passing measurements alternate ±40m (no consensus)."""
        vo = _StubVO(dlon_per_step=0.0001, dlat_per_step=0.0001)
        igf = _make_igf(vo, kf_q=100.0, kf_r=600.0, chi2_gate=5.99,
                        reanchor_after=5, mn_m=3, mn_n=5, consensus_radius=15.0)
        igf.reset()

        lon, lat = _warm_up(igf)

        # Phase 1: drive into re-anchor mode (consec_reject >= reanchor_after=5)
        lon, lat = self._setup_in_reanchor_mode(igf, lon, lat, n_reject=8)

        # Verify we're in re-anchor mode
        # (P has grown large, so 40m alternating offsets will pass the gate, but ±40m
        #  adjacent measurements are 80m apart > consensus_radius=15m → M-of-N blocks)
        off_p = _m_to_dlon(40.0)
        off_n = _m_to_dlon(-40.0)
        false_accepts = 0
        for i in range(10):
            lon += 0.0001
            lat += 0.0001
            offset = off_p if i % 2 == 0 else off_n
            fused, info = _step(igf, lon + offset, lat)
            if info['accept']:
                false_accepts += 1

        self.assertLess(false_accepts, 4,
                        f"Scattered (alternating ±40m) measurements should not trigger M-of-N consensus, "
                        f"got {false_accepts} spurious accepts")


# ---------------------------------------------------------------------------
# (d) Soft gate: 30-50m middle jumps
# ---------------------------------------------------------------------------

class TestScenarioD(unittest.TestCase):

    def _run_with_soft(self, use_soft_gate: bool, soft_knee: float = 1.0):
        vo = _StubVO(dlon_per_step=0.0001, dlat_per_step=0.0001)
        igf = _make_igf(vo, kf_q=100.0, kf_r=600.0, chi2_gate=5.99,
                        use_soft_gate=use_soft_gate, soft_knee=soft_knee)
        igf.reset()

        cos_lat = math.cos(math.radians(35.67))
        mid_offset = 40.0 / (111320.0 * cos_lat)  # ~40m, should pass hard gate

        lon, lat = 139.69, 35.67
        fused_positions = []
        for i in range(20):
            lon += 0.0001
            lat += 0.0001
            model_lon = lon + mid_offset if i >= 5 else lon
            fused, info = _step(igf, model_lon, lat)
            if i >= 5:
                fused_positions.append(fused[0])

        return fused_positions, lon + mid_offset

    def test_soft_gate_reduces_pull(self):
        """With soft gate on, fused should follow the mid-jump model_pred less."""
        fused_off, final_model = self._run_with_soft(use_soft_gate=False)
        fused_on, _ = self._run_with_soft(use_soft_gate=True, soft_knee=1.0)

        # soft gate OFF: large K → fused tracks model_pred → small dist_off
        # soft gate ON:  R_eff inflated → small K → fused stays behind → large dist_on
        dist_off = abs(fused_off[-1] - final_model)
        dist_on = abs(fused_on[-1] - final_model)
        self.assertLess(dist_off, dist_on,
                        "Soft gate off: larger K pulls fused closer to model_pred; "
                        "soft gate on: inflated R_eff reduces K, fused stays farther away")

    def test_soft_gate_inactive_for_low_d2(self):
        """For accurate model (d²≈0.5), soft gate on/off should give identical results."""
        vo_off = _StubVO(dlon_per_step=0.0001, dlat_per_step=0.0001)
        vo_on = _StubVO(dlon_per_step=0.0001, dlat_per_step=0.0001)
        igf_off = _make_igf(vo_off, use_soft_gate=False)
        igf_on = _make_igf(vo_on, use_soft_gate=True, soft_knee=1.0)
        igf_off.reset()
        igf_on.reset()

        lon, lat = 139.69, 35.67
        for _ in range(20):
            lon += 0.0001
            lat += 0.0001
            f_off, _ = _step(igf_off, lon, lat)
            f_on, _ = _step(igf_on, lon, lat)

        dist_m = _haversine_m(f_off[0], f_off[1], f_on[0], f_on[1])
        self.assertLess(dist_m, 1.0,
                        f"Soft gate should not affect accurate measurements (d²≈0.5); diff={dist_m:.3f}m")


# ---------------------------------------------------------------------------
# (e) VO Scale EMA: scale converges toward true value on accepted accurate steps
# ---------------------------------------------------------------------------

class TestScaleEMA(unittest.TestCase):
    """VO underestimates by 50% → scale should converge from 1.0 toward 2.0."""

    def _make_scaled_vo(self, true_scale: float, init_scale: float = 1.0):
        """
        Returns a StubVO whose raw displacement is 0.0001 deg/step,
        but scale_factor starts at init_scale, while true motion is
        0.0001 * (true_scale / init_scale) deg/step from the model's perspective.

        The EMA calibration target: true_step / (raw * current_scale) = true_scale / init_scale
        → after convergence, vo_scale should approach true_scale.
        """
        # Raw VO returns 0.0001 / true_scale deg (underestimate by true_scale)
        raw_step = 0.0001 / true_scale
        vo = _StubVO(dlon_per_step=raw_step * init_scale,  # reported = raw * init_scale
                     dlat_per_step=raw_step * init_scale)
        vo.scale_factor = init_scale
        return vo, raw_step

    def test_scale_converges_upward(self):
        """VO underestimates 2× → vo_scale should grow from 1.0 toward 2.0."""
        true_scale = 2.0
        init_scale = 1.0
        # VO raw step = 0.00005, with scale=1.0 → reports 0.00005 deg/step
        # True motion (model): 0.0001 deg/step
        # Calibration target = 0.0001 / 0.00005 = 2.0
        vo = _StubVO(dlon_per_step=0.00005, dlat_per_step=0.00005)
        igf = _make_igf(vo, use_scale_ema=True, ema_alpha=0.2,
                        vo_scale_init=init_scale, kf_q=100.0, kf_r=600.0)
        igf.reset()

        lon, lat = _warm_up(igf)  # bring P to steady state with correct 0.0001 steps
        # After warm-up, igf vo_scale has been updated. Reset to init to test convergence.
        igf.vo_scale = init_scale
        igf.vo.scale_factor = init_scale

        # Run 50 accurate accepted steps: model at 0.0001/step, VO raw at 0.00005/step
        for _ in range(50):
            lon += 0.0001
            lat += 0.0001
            _step(igf, lon, lat)

        final_scale = igf.vo_scale
        self.assertGreater(final_scale, 1.3,
                           f"Scale should converge upward from 1.0 toward 2.0; got {final_scale:.3f}")

    def test_scale_disabled_stays_fixed(self):
        """With use_scale_ema=False, vo_scale must not change regardless of accepted steps."""
        vo = _StubVO(dlon_per_step=0.00005, dlat_per_step=0.00005)
        igf = _make_igf(vo, use_scale_ema=False, vo_scale_init=1.0)
        igf.reset()

        lon, lat = _warm_up(igf)
        igf.vo_scale = 1.0
        igf.vo.scale_factor = 1.0

        for _ in range(50):
            lon += 0.0001
            lat += 0.0001
            _step(igf, lon, lat)

        self.assertAlmostEqual(igf.vo_scale, 1.0, places=6,
                               msg="vo_scale must not change when use_scale_ema=False")

    def test_ema_updated_flag_set_on_accept(self):
        """info['ema_updated'] should be 1 on accepted steps when use_scale_ema=True."""
        vo = _StubVO(dlon_per_step=0.0001, dlat_per_step=0.0001)
        igf = _make_igf(vo, use_scale_ema=True, ema_alpha=0.1, vo_scale_init=1.0)
        igf.reset()

        lon, lat = _warm_up(igf)
        lon += 0.0001; lat += 0.0001
        fused, info = _step(igf, lon, lat)
        if info['accept']:
            self.assertEqual(info['ema_updated'], 1,
                             "ema_updated should be 1 on accepted step with scale_ema enabled")

    def test_scale_clamped_at_bounds(self):
        """vo_scale is clamped to scale_clamp; extreme miscalibration cannot escape bounds."""
        # VO reports 1000× too little → target_scale would be huge
        vo = _StubVO(dlon_per_step=0.0000001, dlat_per_step=0.0000001)
        igf = _make_igf(vo, use_scale_ema=True, ema_alpha=0.5, vo_scale_init=1.0,
                        scale_clamp=(0.5, 3.0))
        igf.reset()
        lon, lat = _warm_up(igf)

        igf.vo_scale = 1.0
        igf.vo.scale_factor = 1.0
        for _ in range(20):
            lon += 0.0001; lat += 0.0001
            _step(igf, lon, lat)

        self.assertLessEqual(igf.vo_scale, 3.0,
                             f"vo_scale must not exceed clamp upper bound; got {igf.vo_scale:.3f}")


# ---------------------------------------------------------------------------
# (f) Heading limiter: heading rate is clamped near waypoint
# ---------------------------------------------------------------------------

def _apply_heading_limit(prev_angle, new_angle, max_rate):
    """Isolated replica of the heading limiter formula in nav.py:fly()."""
    delta_h = (new_angle - prev_angle + 180.0) % 360.0 - 180.0
    if abs(delta_h) > max_rate:
        return (prev_angle + max_rate * (1.0 if delta_h > 0 else -1.0)) % 360.0
    return new_angle


class TestHeadingLimiter(unittest.TestCase):

    def test_large_cw_turn_clamped(self):
        """90° CW turn from 0° → 90° with max_rate=30° should yield 30°."""
        result = _apply_heading_limit(0.0, 90.0, max_rate=30.0)
        self.assertAlmostEqual(result, 30.0, places=5)

    def test_large_ccw_turn_clamped(self):
        """90° CCW turn: 90° → 0° with max_rate=30° should yield 60°."""
        result = _apply_heading_limit(90.0, 0.0, max_rate=30.0)
        self.assertAlmostEqual(result, 60.0, places=5)

    def test_small_turn_not_clamped(self):
        """10° turn with max_rate=30°: no clamping, angle passes through."""
        result = _apply_heading_limit(0.0, 10.0, max_rate=30.0)
        self.assertAlmostEqual(result, 10.0, places=5)

    def test_exact_limit_not_clamped(self):
        """Turn exactly equal to max_rate: should pass through unchanged."""
        result = _apply_heading_limit(0.0, 30.0, max_rate=30.0)
        self.assertAlmostEqual(result, 30.0, places=5)

    def test_wrap_across_zero(self):
        """350° → 10° is a +20° turn (shortest path), max_rate=30°: no clamp."""
        result = _apply_heading_limit(350.0, 10.0, max_rate=30.0)
        self.assertAlmostEqual(result, 10.0, places=5)

    def test_wrap_across_zero_large(self):
        """350° → 100° is a +110° turn (shortest path); max_rate=30° → should clamp to 20°."""
        result = _apply_heading_limit(350.0, 100.0, max_rate=30.0)
        self.assertAlmostEqual(result, 20.0, places=5)

    def test_wrap_negative_direction(self):
        """10° → 290° is a −80° turn (shortest path); max_rate=30° → clamp to 340°."""
        result = _apply_heading_limit(10.0, 290.0, max_rate=30.0)
        self.assertAlmostEqual(result, 340.0, places=5)

    def test_output_always_in_0_360(self):
        """Clamped output must stay in [0, 360)."""
        result = _apply_heading_limit(10.0, 290.0, max_rate=30.0)
        self.assertGreaterEqual(result, 0.0)
        self.assertLess(result, 360.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

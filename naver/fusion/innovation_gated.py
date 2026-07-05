import math
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

from naver.vo.uav_vo import FrameToFrameVO


# ---------------------------------------------------------------------------
# Geometry helpers (flat-earth approximation, valid for <10 km)
# ---------------------------------------------------------------------------

def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    cos_lat = math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.sqrt(((lat2 - lat1) * 111320) ** 2 +
                     ((lon2 - lon1) * 111320 * cos_lat) ** 2)


def _deg_to_m(dlon: float, dlat: float, ref_lat: float) -> Tuple[float, float]:
    cos_lat = math.cos(math.radians(ref_lat))
    return dlon * 111320.0 * cos_lat, dlat * 111320.0


# ---------------------------------------------------------------------------
# Innovation-Gated Kalman Filter
# ---------------------------------------------------------------------------

class InnovationGatedFusion:
    """
    DR-Bearing v2: Innovation-Gated Kalman Filter position fusion.

    Design rationale vs v1 (fixed-weight blend):
    - Does NOT ask the model "how confident are you?" (PSG entropy was always ~1).
    - Instead measures disagreement (innovation) between VO-propagated state and
      absolute model measurement: large innovation → model is probably wrong → reject.
    - After long rejection, P grows → gate widens → model measurements become easier
      to re-accept, so VO-only navigation is self-limiting (prevents v1's permanent
      VO-lock).
    - "Consistently wrong" models keep producing large innovations regardless of P
      growth, so M-of-N consensus prevents re-anchoring on a drifted model.

    KF state: (lon, lat) position in degrees.
    P, Q, R are in m²; innovation is computed in metres then converted back.
    """

    def __init__(
        self,
        vo: FrameToFrameVO,
        kf_q: float = 100.0,
        kf_r: float = 600.0,
        chi2_gate: float = 5.99,
        use_soft_gate: bool = False,
        soft_knee: float = 1.0,
        reanchor_after: int = 5,
        mn_m: int = 3,
        mn_n: int = 5,
        consensus_radius: float = 15.0,
        vo_scale_init: float = 1.2,
        use_scale_ema: bool = False,
        ema_alpha: float = 0.1,
        scale_clamp: Tuple[float, float] = (0.5, 3.0),
        scale_noise_floor_m: float = 2.0,
        use_adaptive_r: bool = False,
        r_ema_alpha: float = 0.05,
        r_min: float = 100.0,
        r_max: float = 10000.0,
    ) -> None:
        self.vo = vo
        self.kf_q = kf_q
        self.kf_r = kf_r
        self.chi2_gate = chi2_gate
        self.use_soft_gate = use_soft_gate
        self.soft_knee = soft_knee
        self.reanchor_after = reanchor_after
        self.mn_m = mn_m
        self.mn_n = mn_n
        self.consensus_radius = consensus_radius

        # Adaptive R: EMA estimate of model measurement noise (off by default)
        self.use_adaptive_r: bool = use_adaptive_r
        self.r_ema_alpha: float = r_ema_alpha
        self.r_min: float = r_min
        self.r_max: float = r_max
        self._R_hat: float = kf_r       # running R estimate; equals kf_r when adaptive is off
        self._r_warmup_done: bool = False  # True after first acceptance

        # VO online scale: EMA update on accepted steps (off by default, enabled by use_scale_ema)
        self.vo_scale: float = vo_scale_init
        self.use_scale_ema: bool = use_scale_ema
        self.ema_alpha: float = ema_alpha
        self.scale_clamp: Tuple[float, float] = scale_clamp
        self.scale_noise_floor_m: float = scale_noise_floor_m
        # Sync scale into VO module
        self.vo.scale_factor = self.vo_scale

        # KF state
        self._x_lon: Optional[float] = None
        self._x_lat: Optional[float] = None
        self._P: float = kf_r

        # Rejection tracking for M-of-N re-anchor
        self._consec_reject: int = 0
        self._mn_window: List[Tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._x_lon = None
        self._x_lat = None
        self._P = self.kf_r
        self._consec_reject = 0
        self._mn_window = []
        self._R_hat = self.kf_r
        self._r_warmup_done = False
        self.vo.reset()

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def step(
        self,
        model_pred: Tuple[float, float],
        uvp_img: np.ndarray,
        theta_deg: float,
        scale_m_per_px: float,
        step_vector: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Tuple[float, float], Dict[str, Any]]:
        """
        Fuse model absolute prediction with VO dead-reckoning via KF gate.

        Args:
            model_pred:    Absolute position (lon, lat) from regression model.
            uvp_img:       Current UVP patch (BGR uint8) for VO.
            theta_deg:     Heading angle (fly_angle_ccs) for VO un-rotation.
            scale_m_per_px: RSI metres per pixel (latm_per_pixel).
            step_vector:   Fallback (dlon, dlat) displacement when VO fails.

        Returns:
            (fused_point, info_dict)
            fused_point: (lon, lat) KF-fused position.
            info_dict: diagnostic scalars logged to CSV.
        """
        # --- VO estimate ---
        vo_delta = self.vo.estimate(uvp_img, theta_deg, scale_m_per_px)
        self.vo.push(uvp_img, theta_deg)
        vo_valid = vo_delta is not None

        # --- initialise on first step ---
        if self._x_lon is None:
            self._x_lon = model_pred[0]
            self._x_lat = model_pred[1]
            self._P = self.kf_r
            return model_pred, self._make_info(
                d2=0.0, accept=True, K=1.0, R_eff=self.kf_r,
                vo_valid=vo_valid, vo_delta=vo_delta, ema_updated=False,
            )

        # Capture KF state before prediction (needed for EMA scale calibration)
        prev_x_lon = self._x_lon
        prev_x_lat = self._x_lat

        # ------------------------------------------------------------------
        # Step 1 — Predict
        # ------------------------------------------------------------------
        if vo_valid:
            x_lon_pred = self._x_lon + vo_delta[0]
            x_lat_pred = self._x_lat + vo_delta[1]
            P_pred = self._P + self.kf_q
        elif step_vector is not None:
            # Commanded step direction as VO fallback; double Q to reflect extra uncertainty
            x_lon_pred = self._x_lon + step_vector[0]
            x_lat_pred = self._x_lat + step_vector[1]
            P_pred = self._P + 2.0 * self.kf_q
        else:
            x_lon_pred = self._x_lon
            x_lat_pred = self._x_lat
            P_pred = self._P + 2.0 * self.kf_q

        # ------------------------------------------------------------------
        # Step 2 — Innovation in metres (flat-earth)
        # ------------------------------------------------------------------
        innov_dlon = model_pred[0] - x_lon_pred
        innov_dlat = model_pred[1] - x_lat_pred
        innov_mx, innov_my = _deg_to_m(innov_dlon, innov_dlat, x_lat_pred)
        innov_m2 = innov_mx ** 2 + innov_my ** 2
        d2 = innov_m2 / (P_pred + self._R_hat)

        # ------------------------------------------------------------------
        # Step 3 — Gate + Kalman update
        # ------------------------------------------------------------------
        if d2 < self.chi2_gate:
            accept = self._maybe_accept_mn(model_pred)
        else:
            accept = False

        if accept:
            R_eff = self._R_hat
            if self.use_soft_gate and d2 > self.soft_knee:
                R_eff = self._R_hat * (d2 / self.soft_knee)
            K = P_pred / (P_pred + R_eff)
            self._x_lon = x_lon_pred + K * innov_dlon
            self._x_lat = x_lat_pred + K * innov_dlat
            self._P = (1.0 - K) * P_pred
            self._consec_reject = 0
            self._mn_window = []
        else:
            self._x_lon = x_lon_pred
            self._x_lat = x_lat_pred
            self._P = P_pred
            self._consec_reject += 1
            K = 0.0
            R_eff = self._R_hat

        # ------------------------------------------------------------------
        # Step 4b — Adaptive R update
        # ------------------------------------------------------------------
        if self.use_adaptive_r:
            if accept:
                # accepted: use actual innovation as direct sample of model noise
                r_sample = innov_m2
                self._r_warmup_done = True
            elif not self._r_warmup_done:
                # pre-acceptance warmup: use gate boundary as lower-bound proxy
                # to push R_hat upward so the gate widens and allows first acceptance
                r_sample = self.chi2_gate * (P_pred + self._R_hat)
            else:
                r_sample = None
            if r_sample is not None:
                new_r = self.r_ema_alpha * r_sample + (1.0 - self.r_ema_alpha) * self._R_hat
                self._R_hat = max(self.r_min, min(self.r_max, new_r))

        # ------------------------------------------------------------------
        # Step 4 — VO scale EMA (only on accepted steps with valid VO)
        # ------------------------------------------------------------------
        ema_updated = False
        if self.use_scale_ema and accept and vo_valid and self.vo_scale > 0.0:
            # Reference displacement: model_pred relative to previous KF state
            ref_mx, ref_my = _deg_to_m(
                model_pred[0] - prev_x_lon,
                model_pred[1] - prev_x_lat,
                prev_x_lat,
            )
            ref_m = math.sqrt(ref_mx ** 2 + ref_my ** 2)
            # VO-reported displacement (already scaled by current vo_scale)
            vo_mx, vo_my = _deg_to_m(vo_delta[0], vo_delta[1], prev_x_lat)
            vo_m = math.sqrt(vo_mx ** 2 + vo_my ** 2)
            # Only calibrate when both displacements exceed noise floor
            if vo_m > self.scale_noise_floor_m and ref_m > self.scale_noise_floor_m:
                raw_m = vo_m / self.vo_scale  # unscaled VO magnitude
                target_scale = ref_m / raw_m
                new_scale = self.ema_alpha * target_scale + (1.0 - self.ema_alpha) * self.vo_scale
                self.vo_scale = max(self.scale_clamp[0], min(self.scale_clamp[1], new_scale))
                self.vo.scale_factor = self.vo_scale
                ema_updated = True

        fused = (self._x_lon, self._x_lat)
        return fused, self._make_info(
            d2=d2, accept=accept, K=K, R_eff=R_eff,
            vo_valid=vo_valid, vo_delta=vo_delta, ema_updated=ema_updated,
        )

    # ------------------------------------------------------------------
    # M-of-N re-anchor consensus check
    # ------------------------------------------------------------------

    def _maybe_accept_mn(self, model_pred: Tuple[float, float]) -> bool:
        """
        If not in re-anchor mode (consec_reject < reanchor_after), accept immediately.
        Otherwise, require M of the last N gate-passing measurements to cluster within
        consensus_radius before resuming normal Kalman updates.
        """
        if self._consec_reject < self.reanchor_after:
            return True

        # Accumulate gate-passing candidates
        self._mn_window.append(model_pred)
        if len(self._mn_window) > self.mn_n:
            self._mn_window = self._mn_window[-self.mn_n:]

        if len(self._mn_window) < self.mn_m:
            return False

        # Check pairwise distance among the last mn_m candidates
        recent = self._mn_window[-self.mn_m:]
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                d = _haversine_m(recent[i][0], recent[i][1],
                                 recent[j][0], recent[j][1])
                if d > self.consensus_radius:
                    return False
        return True

    # ------------------------------------------------------------------
    # Info dict helper
    # ------------------------------------------------------------------

    def _make_info(
        self,
        d2: float,
        accept: bool,
        K: float,
        R_eff: float,
        vo_valid: bool,
        vo_delta: Optional[Tuple[float, float]],
        ema_updated: bool = False,
    ) -> Dict[str, Any]:
        return {
            'd2': d2,
            'accept': int(accept),
            'K': K,
            'P': self._P,
            'R_eff': R_eff,
            'R_hat': self._R_hat,
            'consec_reject': self._consec_reject,
            'mn_window_len': len(self._mn_window),
            'vo_valid': int(vo_valid),
            'vo_scale': self.vo_scale,
            'vo_delta_lon': vo_delta[0] if vo_delta else None,
            'vo_delta_lat': vo_delta[1] if vo_delta else None,
            'ema_updated': int(ema_updated),
        }

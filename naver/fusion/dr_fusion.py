import math
from typing import Optional, Tuple

import cv2
import numpy as np

from naver.vo.uav_vo import FrameToFrameVO


def _haversine_m(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Approximate distance in metres between two (lon, lat) points."""
    dlat = math.radians(p2[1] - p1[1])
    dlon = math.radians(p2[0] - p1[0])
    return math.sqrt((dlat * 111320) ** 2 + (dlon * 111320 * math.cos(math.radians(p1[1]))) ** 2)


class DRBearingFusion:
    """
    DR-Bearing Phase 3: fuse absolute position regression with VO dead reckoning.

    At each step:
      - model_pred  = PARCASGM_v5 position estimate (absolute)
      - vo_pred     = prev_fused + VO_delta (incremental, independent of model)
      - fused       = (1-u)*model_pred + u*vo_pred

    u ∈ [0,1] is derived from PSG α entropy: u≈0 → trust model, u≈1 → trust VO.

    Sanity gate: if VO estimate is farther than anchor_threshold metres from the
    model estimate, the VO is discarded for that step (model prediction is kept).
    This prevents VO drift from hijacking navigation when RANSAC fails silently.
    """

    def __init__(
        self,
        vo: FrameToFrameVO,
        anchor_threshold: float = 30.0,
    ) -> None:
        self.vo = vo
        self.anchor_threshold = anchor_threshold
        self._prev_fused: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self._prev_fused = None
        self.vo.reset()

    def step(
        self,
        model_pred: Tuple[float, float],
        uvp_img: np.ndarray,
        theta_deg: float,
        scale_m_per_px: float,
        u: float,
    ) -> Tuple[Tuple[float, float], bool]:
        """
        Compute fused position estimate for the current step.

        Args:
            model_pred:     Absolute model prediction (lon, lat).
            uvp_img:        Current UVP image (256×256 BGR uint8).
            theta_deg:      fly_angle_ccs at this step.
            scale_m_per_px: latm_per_pixel (metres per pixel).
            u:              Uncertainty in [0,1] from entropy_from_alpha.

        Returns:
            (fused_point, vo_used): fused (lon, lat) and whether VO contributed.
        """
        vo_delta = self.vo.estimate(uvp_img, theta_deg, scale_m_per_px)
        self.vo.push(uvp_img, theta_deg)

        if vo_delta is None or self._prev_fused is None:
            self._prev_fused = model_pred
            return model_pred, False

        vo_pred = (
            self._prev_fused[0] + vo_delta[0],
            self._prev_fused[1] + vo_delta[1],
        )

        dist = _haversine_m(model_pred, vo_pred)
        if dist > self.anchor_threshold:
            self._prev_fused = model_pred
            return model_pred, False

        fused = (
            (1.0 - u) * model_pred[0] + u * vo_pred[0],
            (1.0 - u) * model_pred[1] + u * vo_pred[1],
        )
        self._prev_fused = fused
        return fused, True

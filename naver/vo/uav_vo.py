import math
from typing import Optional, Tuple

import cv2
import numpy as np


class FrameToFrameVO:
    """
    Frame-to-frame Visual Odometry using phase correlation on un-rotated RSI patches.

    UVP images are saved after rotating the RSI clockwise by theta (fly_angle_ccs).
    To compare consecutive frames that may have different headings, we first undo
    the rotation (rotate CCW by theta) to restore them to the canonical RSI frame
    (north-up, east-right, y increasing south).

    Then cv2.phaseCorrelate gives the translation (sx, sy) in RSI pixel coordinates:
        delta_lon = -sx * scale_deg   (sx > 0 → features moved east → drone moved west: sign flip)
        delta_lat =  sy * scale_deg   (sy > 0 → features moved south → drone moved north: sign flip)

    Wait: carefully, phaseCorrelate returns (sx, sy) such that img2 ≈ shift(img1, sx, sy).
    If drone moved east by D pixels, features in RSI_new are D pixels to the LEFT of RSI_old.
    img1 (old RSI) shifted right by D matches img2 (new RSI): sx = -D < 0.
    So: delta_lon = -sx * scale_deg (positive when drone moved east). ✓

    If drone moved north, features in RSI_new are D pixels BELOW RSI_old (y-down, north=small y).
    img1 shifted down by D matches img2: sy = +D > 0.
    delta_lat = sy * scale_deg (positive when drone moved north). ✓

    Phase correlation systematically underestimates ~80% of true displacement on urban patches.
    A calibration factor `scale_factor=1.25` corrects this empirically.
    """

    def __init__(
        self,
        min_response: float = 0.05,
        scale_factor: float = 1.25,
    ) -> None:
        """
        Args:
            min_response:  Minimum phase-correlation peak response to accept a result.
            scale_factor:  Empirical correction for phase-correlation magnitude bias (~80% underestimate on urban RSI).
        """
        self.min_response = min_response
        self.scale_factor = scale_factor
        self._prev_img: Optional[np.ndarray] = None
        self._prev_theta: Optional[float] = None

    def reset(self) -> None:
        self._prev_img = None
        self._prev_theta = None

    def push(self, img: np.ndarray, theta_deg: float) -> None:
        self._prev_img = img.copy()
        self._prev_theta = theta_deg

    def estimate(
        self,
        curr_img: np.ndarray,
        curr_theta: float,
        scale_m_per_px: float,
    ) -> Optional[Tuple[float, float]]:
        """
        Estimate (delta_lon, delta_lat) in degrees between previous and current UVP.

        Args:
            curr_img:       Current UVP image (256×256, BGR uint8).
            curr_theta:     fly_angle_ccs at this step (degrees, CCW from east).
            scale_m_per_px: latm_per_pixel (metres per RSI pixel).

        Returns:
            (delta_lon, delta_lat) in degrees, or None if VO quality is below threshold.
        """
        if self._prev_img is None or self._prev_theta is None:
            return None

        prev_rsi = self._unrotate(self._prev_img, self._prev_theta)
        curr_rsi = self._unrotate(curr_img, curr_theta)

        prev_f = self._to_gray_float(prev_rsi)
        curr_f = self._to_gray_float(curr_rsi)

        shift, response = cv2.phaseCorrelate(prev_f, curr_f)

        if response < self.min_response:
            return None

        sx, sy = float(shift[0]), float(shift[1])
        scale_deg = scale_m_per_px / 111320.0

        delta_lon = -sx * scale_deg * self.scale_factor
        delta_lat =  sy * scale_deg * self.scale_factor

        return (delta_lon, delta_lat)

    @staticmethod
    def _unrotate(img: np.ndarray, theta_deg: float) -> np.ndarray:
        """Undo the CW-by-theta rotation: apply CCW by theta."""
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), theta_deg, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def _to_gray_float(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img.astype(np.float32)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

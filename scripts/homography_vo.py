"""
Homography-based relative camera motion (R, t) between two drone/aerial UVP frames.

Pipeline: ORB features -> BFMatcher(Hamming) + Lowe's ratio test -> RANSAC
homography -> decomposeHomographyMat -> pick the physically valid solution
(positive-depth / visible-refpoints filter) -> report R as roll/pitch/yaw and
t as a direction (+ an approximate metric rescale using known altitude).

IMPORTANT caveat (see decomposeHomographyMat's own docs): a homography only
recovers translation *direction*, normalized by the (unknown) scene depth --
not metric distance -- unless you separately know that depth. This script
still reports a metres estimate, but only by borrowing altitude/GSD from this
project's known imaging geometry to rescale it; that rescale is only as good
as "the ground under these two crops is roughly flat," which is exactly the
assumption 3d/UAV-view parallax breaks.

Usage:
    conda activate bearing_env
    python scripts/homography_vo.py <img1> <img2> [--out matches.jpg]
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1) Load
# ---------------------------------------------------------------------------

def load_gray(path: str):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


# ---------------------------------------------------------------------------
# 2+3) ORB features, BFMatcher + Lowe's ratio test
# ---------------------------------------------------------------------------

def match_features(gray1: np.ndarray, gray2: np.ndarray, n_features: int = 3000, ratio: float = 0.75):
    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return kp1, kp2, []

    # crossCheck=False is required for knnMatch (crossCheck only supports k=1)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:  # Lowe's ratio test
            good.append(m)
    return kp1, kp2, good


# ---------------------------------------------------------------------------
# 4) RANSAC homography + match visualization
# ---------------------------------------------------------------------------

def estimate_homography(kp1, kp2, good_matches, ransac_thresh: float = 5.0):
    if len(good_matches) < 4:
        return None, None, None, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    return H, mask, pts1, pts2


def save_match_visualization(img1, kp1, img2, kp2, good_matches, mask, out_path: str):
    mask_flat = mask.ravel().tolist() if mask is not None else None
    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=mask_flat,  # only draw RANSAC inliers
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    vis = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
    cv2.imwrite(out_path, vis)
    return vis


# ---------------------------------------------------------------------------
# 5) Camera intrinsics + homography decomposition + solution selection
# ---------------------------------------------------------------------------

def build_intrinsics(image_width: int, image_height: int, altitude_m: float, gsd_m_per_px: float) -> np.ndarray:
    """
    Pinhole K for a near-nadir aerial/drone camera, derived from imaging
    geometry we actually know (altitude + ground sample distance) rather than
    an arbitrary guess: for a nadir view, focal_length_px ~= altitude_m / gsd_m_per_px
    (similar-triangles: a ground patch gsd_m_per_px metres wide subtends 1 px
    at range altitude_m).
    """
    f_px = altitude_m / gsd_m_per_px
    cx, cy = image_width / 2.0, image_height / 2.0
    K = np.array([
        [f_px, 0.0, cx],
        [0.0, f_px, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K


def decompose_and_select(H, K, pts1, pts2, mask):
    """
    decomposeHomographyMat returns up to 4 mathematically-valid (R, t, n)
    solutions. filterHomographyDecompByVisibleRefpoints narrows this using the
    actual inlier correspondences (points must stay in front of both cameras).
    If more than one candidate still survives, we break the tie with a simple
    domain heuristic: for a near-nadir aerial camera, the ground-plane normal
    should point roughly back along the camera boresight (n ~= [0,0,-1] or
    [0,0,1] depending on convention) -- pick the surviving solution whose
    normal is most aligned with that.
    """
    n_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)

    inlier_mask = mask.ravel().astype(bool)
    inlier_pts1 = pts1[inlier_mask]
    inlier_pts2 = pts2[inlier_mask]

    possible = cv2.filterHomographyDecompByVisibleRefpoints(
        rotations, normals, inlier_pts1, inlier_pts2
    )
    if possible is None or len(possible) == 0:
        # positive-depth filter rejected everything (degenerate H / too few inliers);
        # fall back to considering all raw candidates rather than crashing.
        possible = np.arange(n_solutions)
    else:
        possible = possible.ravel()

    boresight = np.array([0.0, 0.0, 1.0])
    best_idx = min(possible, key=lambda i: 1.0 - abs(float(normals[i].ravel() @ boresight)))
    return rotations[best_idx], translations[best_idx], normals[best_idx], int(best_idx), int(n_solutions)


# ---------------------------------------------------------------------------
# 6) Report
# ---------------------------------------------------------------------------

def rotation_matrix_to_euler_deg(R: np.ndarray):
    """Standard R = Rz(yaw) @ Ry(pitch) @ Rx(roll) decomposition, in degrees."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return tuple(math.degrees(a) for a in (roll, pitch, yaw))


def run(img1_path: str, img2_path: str, altitude_m: float, gsd_m_per_px: float,
        out_path: str = None, n_features: int = 3000, ratio: float = 0.75, ransac_thresh: float = 5.0):
    img1, gray1 = load_gray(img1_path)
    img2, gray2 = load_gray(img2_path)
    h, w = gray1.shape[:2]

    kp1, kp2, good = match_features(gray1, gray2, n_features=n_features, ratio=ratio)
    print(f"ORB keypoints: {len(kp1)} / {len(kp2)}  |  ratio-test survivors: {len(good)}")

    H, mask, pts1, pts2 = estimate_homography(kp1, kp2, good, ransac_thresh=ransac_thresh)
    if H is None:
        print("FAIL: not enough ratio-test matches to fit a homography (need >= 4).")
        return None

    n_inliers = int(mask.sum())
    print(f"RANSAC homography inliers: {n_inliers} / {len(good)} ({100.0 * n_inliers / len(good):.0f}%)")

    if out_path:
        save_match_visualization(img1, kp1, img2, kp2, good, mask, out_path)
        print(f"Match visualization saved to {out_path}")

    K = build_intrinsics(w, h, altitude_m, gsd_m_per_px)
    print(f"K (focal_px={K[0,0]:.1f}, derived from altitude={altitude_m:.1f} m / gsd={gsd_m_per_px} m/px):\n{K}")

    R, t, n, chosen_idx, n_solutions = decompose_and_select(H, K, pts1, pts2, mask)
    roll, pitch, yaw = rotation_matrix_to_euler_deg(R)

    t_dir = (t / (np.linalg.norm(t) + 1e-12)).ravel()
    # SCALE CAVEAT: t from decomposeHomographyMat is translation / plane_depth.
    # Rescaling by the known altitude is only valid if the imaged ground is
    # ~flat and near that altitude below the camera -- exactly the assumption
    # that breaks under 3d/UAV-view parallax (tall structures, oblique angle).
    t_metric_approx = t_dir * altitude_m

    print(f"\nSelected solution {chosen_idx + 1} of {n_solutions} candidates (positive-depth + boresight-normal filter)")
    print(f"Rotation (deg):  roll={roll:+.2f}  pitch={pitch:+.2f}  yaw={yaw:+.2f}")
    print(f"Plane normal (camera frame): {n.ravel()}")
    print(f"Translation direction (unit, scale-ambiguous): {t_dir}")
    print(f"Translation, approx. metres (rescaled by altitude={altitude_m:.1f} m -- "
          f"ONLY valid if ground under the crop is flat and near that altitude): {t_metric_approx}")
    print("NOTE: decomposeHomographyMat cannot recover metric scale on its own -- "
          "the metres figure above is a rescale assumption, not a measurement.")

    return {
        "R": R, "t_dir": t_dir, "t_metric_approx": t_metric_approx,
        "roll": roll, "pitch": pitch, "yaw": yaw,
        "n_inliers": n_inliers, "n_good": len(good),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("img1")
    ap.add_argument("img2")
    ap.add_argument("--altitude_m", type=float, default=204.5257,
                     help="known flight altitude in metres (default: this project's 254k rsi_type)")
    ap.add_argument("--gsd", type=float, default=0.25, help="metres per pixel (default: this project's 254k patches)")
    ap.add_argument("--out", default=None, help="path to save the RANSAC-inlier match visualization")
    ap.add_argument("--n_features", type=int, default=3000)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac_thresh", type=float, default=5.0)
    args = ap.parse_args()
    run(args.img1, args.img2, args.altitude_m, args.gsd, args.out, args.n_features, args.ratio, args.ransac_thresh)


if __name__ == "__main__":
    main()

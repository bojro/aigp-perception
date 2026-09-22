"""Gate pose from the eight keypoints, standalone.

The flight repo already solves this in ``vision/yolo_pnp.py`` and that is the
copy to use when it is importable. This one exists so the detector can be
verified on a machine that has only this repo, and so the constants that
matter are written down in one place where they can be argued with.

The gate is planar: a 2.7 m outer square with a 1.5 m opening in it, and the
keypoints are indexed by identity rather than position -- 0-3 the outer ring,
4-7 the inner, each clockwise from top-left. That indexing is what lets a gate
which is half out of frame still solve, because any four surviving points
still know which four they are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

GATE_OUTER_M = 2.7
GATE_INNER_M = 1.5
_HO, _HI = GATE_OUTER_M / 2.0, GATE_INNER_M / 2.0

OBJECT_POINTS = np.array([
    [-_HO, -_HO, 0.0], [+_HO, -_HO, 0.0], [+_HO, +_HO, 0.0], [-_HO, +_HO, 0.0],
    [-_HI, -_HI, 0.0], [+_HI, -_HI, 0.0], [+_HI, +_HI, 0.0], [-_HI, +_HI, 0.0],
], dtype=np.float64)

# On-site ChArUco, measured at 1920x1080. Scaled to whatever frame arrives.
CHARUCO_WH = (1920.0, 1080.0)
CHARUCO_K = np.array([[1302.941063, 0.0, 952.287144],
                      [0.0, 1303.105865, 529.035788],
                      [0.0, 0.0, 1.0]], dtype=np.float64)

MIN_POINTS = 4               # planar PnP needs four; a cut gate is the norm
MIN_SPREAD_PX = 12.0         # a quad this small is noise pretending to be a gate
MIN_RANGE_M, MAX_RANGE_M = 0.8, 45.0
MIN_KEYPOINT_CONF = 0.25     # matches vision.dual_gate_pnp

# Reprojection ceiling, in pixels of the 640x360 stream. The flight repo ships
# 2.0, which is right for the simulator's exact corners and far too tight for a
# real lens. Measured over 1207 real frames, 2.0 px throws away 74% of the
# gates this model finds where 8.0 px throws away 33%; and what survives the
# looser cap is no jumpier, in fact slightly steadier -- 16.3% of consecutive
# range steps exceed 1.5 m at 2 px against 11.8% at 8 px. The cap was
# discarding good solves, not bad ones. Raise it further only with evidence:
# the jitter curve stays flat out to 20 px, so the ceiling is not what is
# holding accuracy, but nothing here has yet been checked against a surveyed
# gate distance.
REPROJ_ERR_MAX_PX = 8.0


@dataclass
class GatePose:
    R_cg: np.ndarray             # gate -> camera-optical rotation
    t_cg: np.ndarray             # gate centre in the camera frame (m)
    reproj_err_px: float
    keypoint_ids: tuple
    centre_px: tuple             # gate centre projected back into the frame
    scale: float                 # px per stream-px, for reporting

    @property
    def range_m(self) -> float:
        return float(np.linalg.norm(self.t_cg))

    @property
    def points_used(self) -> int:
        return len(self.keypoint_ids)


def _upright_and_ahead(R: np.ndarray) -> bool:
    """Gates hang the right way up, which resolves IPPE's mirrored pair."""
    return bool(R[1, 1] > 0 and R[2, 2] > 0)


def _solve(obj: np.ndarray, img: np.ndarray, K: np.ndarray):
    o = np.ascontiguousarray(obj.reshape(-1, 1, 3), np.float64)
    p = np.ascontiguousarray(img.reshape(-1, 1, 2), np.float64)

    def attempt(flag):
        try:
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(o, p, K, None, flags=flag)
        except cv2.error:
            return []
        if not ok:
            return []
        # IPPE fits a homography above four points and returns NaN on a
        # near fronto-parallel gate, which is the view on a straight approach.
        return [(r, t) for r, t in zip(rvecs, tvecs)
                if np.isfinite(r).all() and np.isfinite(t).all()]

    pairs = attempt(cv2.SOLVEPNP_IPPE) or attempt(cv2.SOLVEPNP_SQPNP)
    best, best_err = None, REPROJ_ERR_MAX_PX
    for rvec, tvec in pairs:
        if not (MIN_RANGE_M < float(np.asarray(tvec).ravel()[2]) < MAX_RANGE_M):
            continue
        R, _ = cv2.Rodrigues(rvec)
        if not _upright_and_ahead(R):
            continue
        proj, _ = cv2.projectPoints(o, rvec, tvec, K, None)
        err = float(np.sqrt(((proj.reshape(-1, 2) - img) ** 2).sum(axis=1).mean()))
        if err < best_err:
            best, best_err = (R, np.asarray(tvec).reshape(3), rvec), err
    return (best, best_err) if best else (None, float("inf"))


def solve_gate_pose(keypoints_px, keypoint_conf, frame_shape,
                    confidence: float = 1.0, bbox=None,
                    min_keypoint_conf: float = MIN_KEYPOINT_CONF,
                    K: Optional[np.ndarray] = None) -> Optional[GatePose]:
    """Pose from whatever keypoints survived, or None if too few did.

    Matches the signature ``GateDetector`` injects, so it can be passed
    straight in as ``pose_solver``.
    """
    pts = np.asarray(keypoints_px, np.float64).reshape(-1, 2)
    conf = np.asarray(keypoint_conf, np.float64).reshape(-1)
    if pts.shape[0] != len(OBJECT_POINTS) or conf.shape[0] != pts.shape[0]:
        return None

    h, w = frame_shape[:2]
    if K is None:
        # The intrinsics were measured at 1920x1080; rescale, do not assume.
        K = CHARUCO_K.copy()
        K[0, :] *= w / CHARUCO_WH[0]
        K[1, :] *= h / CHARUCO_WH[1]

    usable = (np.isfinite(pts).all(axis=1) & np.isfinite(conf)
              & (conf >= min_keypoint_conf) & ~np.all(pts == 0.0, axis=1))
    ids = tuple(int(i) for i in np.nonzero(usable)[0])
    if len(ids) < MIN_POINTS:
        return None
    img = pts[list(ids)]
    if max(np.ptp(img[:, 0]), np.ptp(img[:, 1])) < MIN_SPREAD_PX:
        return None

    # The reprojection ceiling is quoted in 640x360 stream pixels, so the
    # comparison is made there whatever the frame that arrived.
    s = 640.0 / w
    best, err = _solve(OBJECT_POINTS[list(ids)], img * s,
                       np.diag([s, s, 1.0]) @ K)
    if best is None:
        return None
    R, t, rvec = best
    centre, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, t.reshape(3, 1),
                                  np.diag([s, s, 1.0]) @ K, None)
    return GatePose(R_cg=R, t_cg=t, reproj_err_px=err, keypoint_ids=ids,
                    centre_px=(float(centre[0, 0, 0]) / s,
                               float(centre[0, 0, 1]) / s), scale=s)

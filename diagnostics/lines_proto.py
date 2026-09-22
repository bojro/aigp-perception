"""Prototype: recover a clipped gate's quad by fitting its edges as lines.

A quadrilateral running off the image has no four contour corners to find, but
its EDGES are still visible. Fit a line to each edge and intersect them and the
corners come back - including the ones outside the frame.
"""
import sys
from pathlib import Path
import math
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp_perception import paths as P
from aigp_perception.autolabel_gate_pose import AutolabelConfig, orange_mask, order_corners

SP = str(P.WORK)
SRC = str(P.CAPTURE)


def edge_segments(contour, shape, border_margin=4, min_frac=0.06):
    """Straight runs of the contour that are real gate edge, not image border."""
    height, width = shape[:2]
    perimeter = cv2.arcLength(contour, True)
    poly = cv2.approxPolyDP(contour, 0.004 * perimeter, True).reshape(-1, 2)
    scale = math.sqrt(max(cv2.contourArea(contour), 1.0))
    out = []
    for i in range(len(poly)):
        a, b = poly[i].astype(float), poly[(i + 1) % len(poly)].astype(float)
        length = np.linalg.norm(b - a)
        if length < min_frac * scale:
            continue
        # A run hugging the image edge is where the frame cut the gate, not a gate edge.
        mid = (a + b) / 2
        on_border = (
            min(a[0], b[0]) <= border_margin and mid[0] <= border_margin
            or min(a[1], b[1]) <= border_margin and mid[1] <= border_margin
            or max(a[0], b[0]) >= width - 1 - border_margin and mid[0] >= width - 1 - border_margin
            or max(a[1], b[1]) >= height - 1 - border_margin and mid[1] >= height - 1 - border_margin
        )
        if on_border:
            continue
        out.append((a, b, length, math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0))
    return out


def intersect(l1, l2):
    (a1, b1), (a2, b2) = l1, l2
    d1, d2 = b1 - a1, b2 - a2
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-9:
        return None
    t = ((a2[0] - a1[0]) * d2[1] - (a2[1] - a1[1]) * d2[0]) / cross
    return a1 + t * d1


def quad_from_lines(contour, shape):
    segments = edge_segments(contour, shape)
    if len(segments) < 4:
        return None
    # Two edge families ~90 deg apart. Seed on the longest segment's angle.
    segments.sort(key=lambda s: -s[2])
    base = segments[0][3]
    fam_a, fam_b = [], []
    for seg in segments:
        delta = min(abs(seg[3] - base), 180 - abs(seg[3] - base))
        (fam_a if delta < 45 else fam_b).append(seg)
    if len(fam_a) < 2 or len(fam_b) < 2:
        return None

    center = contour.reshape(-1, 2).mean(axis=0)
    lines = []
    for family in (fam_a, fam_b):
        fitted = []
        for a, b, length, _ in family:
            direction = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            normal = np.array([-direction[1], direction[0]])
            fitted.append((float(np.dot(center - a, normal)), a, b, length))
        # The two extreme edges of this family are the opposite sides of the quad.
        fitted.sort(key=lambda f: f[0])
        lines += [(fitted[0][1], fitted[0][2]), (fitted[-1][1], fitted[-1][2])]

    corners = []
    for i in (0, 1):
        for j in (2, 3):
            point = intersect(lines[i], lines[j])
            if point is None:
                return None
            corners.append(point)
    return order_corners(np.array(corners, np.float32))


if __name__ == "__main__":
    cfg = AutolabelConfig()
    stems = ["0919_214639_000102", "0919_214639_000210", "0919_214639_000390",
             "0919_214639_000474", "0919_220524_000054", "0919_220524_000282"]
    for stem in stems:
        img = cv2.imread(f"{SRC}/{stem}.jpg")
        mask = orange_mask(img, cfg)
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        top = max((i for i in range(len(cnts)) if hier[0][i][3] == -1),
                  key=lambda i: cv2.contourArea(cnts[i]))
        quad = quad_from_lines(cnts[top], img.shape)
        vis = img.copy()
        vis[mask > 0] = (0.65 * vis[mask > 0] + 0.35 * np.array([0, 140, 255])).astype(np.uint8)
        if quad is None:
            print(f"{stem}: FAIL")
        else:
            side = np.mean([np.linalg.norm(quad[i] - quad[(i+1) % 4]) for i in range(4)])
            off = sum(1 for p in quad if not (0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]))
            print(f"{stem}: side={side:6.0f}px  corners_off_frame={off}  {quad.astype(int).tolist()}")
            cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 0), 5)
            for k, p in enumerate(quad):
                cv2.circle(vis, tuple(p.astype(int)), 14, (255, 0, 255), -1)
                cv2.putText(vis, str(k), tuple(p.astype(int) + 18), 0, 1.4, (255,255,255), 3)
        cv2.imwrite(f"{SP}/lines_{stem}.jpg", cv2.resize(vis, (640, 360)))

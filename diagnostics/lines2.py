"""Clipped-gate quad recovery, take two: convex hull then line intersection.

The convex hull of the orange region is the outer square and nothing else - the
aperture and the hexagon dots are interior, so the hull steps straight over
them. Hull edges that hug the image border are where the frame cut the gate;
drop those and the survivors are real gate sides. Fit each, intersect
consecutive pairs, and corners outside the image come back too.
"""
import sys
from pathlib import Path, math
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp_perception.autolabel_gate_pose import AutolabelConfig, orange_mask, order_corners

SP = str(Path(__file__).resolve().parents[1] / "work")
SRC = "/Users/bojro/Downloads/gate frames"


def _on_border(a, b, shape, margin=5):
    height, width = shape[:2]
    for lo, axis in ((margin, 0), (margin, 1)):
        if a[axis] <= lo and b[axis] <= lo:
            return True
    if a[0] >= width - 1 - margin and b[0] >= width - 1 - margin:
        return True
    if a[1] >= height - 1 - margin and b[1] >= height - 1 - margin:
        return True
    return False


def _fit_line(points):
    """Total-least-squares line through points -> (origin, direction)."""
    vx, vy, x0, y0 = cv2.fitLine(np.asarray(points, np.float32),
                                 cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def _intersect(l1, l2):
    (a1, d1), (a2, d2) = l1, l2
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-9:
        return None
    t = ((a2[0] - a1[0]) * d2[1] - (a2[1] - a1[1]) * d2[0]) / cross
    return a1 + t * d1


def quad_from_hull(contour, shape, angle_tol=22.0, min_frac=0.07):
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(float)
    if len(hull) < 3:
        return None, "hull_degenerate"
    scale = math.sqrt(max(cv2.contourArea(contour), 1.0))

    edges = []   # (points, angle) in hull order, border runs dropped
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        if np.linalg.norm(b - a) < min_frac * scale:
            continue
        if _on_border(a, b, shape):
            continue
        edges.append(([a, b], math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0))
    if len(edges) < 4:
        return None, f"only_{len(edges)}_edges"

    # Merge consecutive hull edges that are really one side broken into pieces.
    groups = [[edges[0]]]
    for edge in edges[1:]:
        delta = abs(edge[1] - groups[-1][-1][1])
        if min(delta, 180 - delta) < angle_tol:
            groups[-1].append(edge)
        else:
            groups.append([edge])
    if len(groups) > 1:
        delta = abs(groups[0][0][1] - groups[-1][-1][1])
        if min(delta, 180 - delta) < angle_tol:   # hull wraps around
            groups[0] = groups[-1] + groups[0]
            groups.pop()
    if len(groups) != 4:
        return None, f"{len(groups)}_sides"

    lines = [_fit_line([p for edge in group for p in edge[0]]) for group in groups]
    corners = []
    for i in range(4):
        point = _intersect(lines[i], lines[(i + 1) % 4])
        if point is None:
            return None, "parallel_sides"
        corners.append(point)
    return order_corners(np.array(corners, np.float32)), "ok"


if __name__ == "__main__":
    cfg = AutolabelConfig()
    stems = ["0919_214639_000102", "0919_214639_000210", "0919_214639_000390",
             "0919_214639_000474", "0919_220524_000054", "0919_220524_000282",
             "0919_214639_000073", "0919_214639_000087"]
    for stem in stems:
        img = cv2.imread(f"{SRC}/{stem}.jpg")
        mask = orange_mask(img, cfg)
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        top = max((i for i in range(len(cnts)) if hier[0][i][3] == -1),
                  key=lambda i: cv2.contourArea(cnts[i]))
        quad, why = quad_from_hull(cnts[top], img.shape)
        vis = img.copy()
        vis[mask > 0] = (0.65 * vis[mask > 0] + 0.35 * np.array([0, 140, 255])).astype(np.uint8)
        if quad is None:
            print(f"{stem}: FAIL ({why})")
        else:
            side = np.mean([np.linalg.norm(quad[i] - quad[(i+1) % 4]) for i in range(4)])
            off = sum(1 for p in quad if not (0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]))
            print(f"{stem}: side={side:6.0f}px off_frame={off}  {quad.astype(int).tolist()}")
            cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 0), 5)
            for k, p in enumerate(quad):
                q = np.clip(p.astype(int), [12, 12], [img.shape[1]-12, img.shape[0]-12])
                cv2.circle(vis, tuple(q), 14, (255, 0, 255), -1)
                cv2.putText(vis, str(k), tuple(q + 18), 0, 1.4, (255, 255, 255), 3)
        cv2.imwrite(f"{SP}/hull_{stem}.jpg", cv2.resize(vis, (640, 360)))

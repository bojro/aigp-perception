"""Auto-label the eight gate keypoints from orange ring topology.

The gate is a square annulus, so a colour mask of it has a *hole*. That single
fact does the work an annotator would otherwise do by hand: the parent contour
is the outer square (ids 0-3) and its child hole is the flyable opening
(ids 4-7). Ring identity therefore comes from topology, never from guessing
which of eight similar corners is which.

Colour thresholding runs deliberately wide - false candidates are cheap because
known geometry rejects them. The gate is planar with a fixed 1500/2700 inner
ratio (spec 3.7), so one homography fitted to the outer quad predicts where the
opening must land. Candidates whose opening misses that prediction are not
gates. That residual, normalised by gate size, is also the review score.

    python tools/autolabel_gate_pose.py "~/Downloads/gate frames" --out datasets/autolabel

Close gates break that topology: the outer square runs off the frame, and often
the opening does too. So the ring is only an *anchor*. Whichever ring is cleanly
observed fixes the homography to the canonical front face, and all eight points
follow from it. Corners that land outside the image are written unseen rather
than invented - a projection four hundred pixels off-frame is a guess, and a
guess that reaches PnP is worse than an honest gap.

Writes YOLOv8-pose labels, a data.yaml that is correct in the three ways a
Roboflow export usually is not (see models/README.md), review overlays, and a
report.csv ranking every instance by how much it needs human eyes.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Spec 3.7: outer boundary 2700 mm, inner square 1500 mm. Both on the front face.
GATE_OUTER_M = 2.7
GATE_INNER_M = 1.5
INNER_RATIO = GATE_INNER_M / GATE_OUTER_M

KEYPOINT_COUNT = 8
# Mirroring swaps TL<->TR and BR<->BL inside each clockwise ring (models/README.md).
FLIP_INDEX = [1, 0, 3, 2, 5, 4, 7, 6]

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

# How much of each side gets probed. Sampling only the middle was quietly
# fatal for clipped gates: when the frame cuts a side, the part still visible
# is usually near one END of it, so the middle-60% window measured nothing and
# the corner went unclaimed even though it sat in plain view. Stations falling
# outside the image are skipped anyway, so widening costs a whole gate nothing.
SIDE_SPAN_LO, SIDE_SPAN_HI = 0.06, 0.94

# Canonical front face, clockwise from top-left, centred on the origin.
CANONICAL_OUTER = np.array(
    [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], np.float32
)
CANONICAL_INNER = (CANONICAL_OUTER * INNER_RATIO).astype(np.float32)


@dataclass(frozen=True)
class AutolabelConfig:
    """Wide colour net, strict geometry. Tune the net, not the gate."""

    # Safety orange under mixed hangar light: dark maroon at range, washed out
    # under the lamps. The sim values in vision/gate_detector.py are far too
    # tight for real frames - they cost ~4x recall on this capture.
    # Swept against the ring-agreement metric on real frames. Tighter than it
    # looks safe: a cleaner mask gives better seed quads, so this raised
    # accuracy AND recall together. Still far wider than the sim values.
    hsv_ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
        ((0, 80, 60), (18, 255, 255)),
    )
    # Swept on real frames: a light close beats a heavy one. Heavy closing
    # drags the mask boundary off the true edge and costs ~3px of accuracy.
    close_kernel: int = 3
    close_iterations: int = 2
    open_kernel: int = 3

    minimum_outer_side_px: float = 60.0   # below this, corners are not labelable
    # A front-on gate's opening covers (1500/2700)^2 = 0.309 of the outer square.
    # Perspective shrinks that; the white hexagon dots sit far below it. These
    # bounds keep the aperture and drop the decoration.
    minimum_hole_area_fraction: float = 0.10
    maximum_hole_area_fraction: float = 0.60

    # Opening-prediction residual as a fraction of outer side length.
    # Orange covering this much of the frame with no usable label means a gate
    # is there and we failed on it - quarantine, never call it a negative.
    unlabelable_area_fraction: float = 0.045

    # Set from the synthetic-crop measurement, not from taste: a corner carried
    # off the frame is wrong by ~6% of gate side at 0.15 of a side, ~17% by
    # 0.30 and ~44% by 0.50. So 0.15 is where a carried corner stops being
    # worth writing down at all.
    trust_reach: float = 0.15

    # What a projected (never measured) ring must score to count as confirmed.
    confirm_support: float = 0.80
    # Correlation with the learned gate face below which a candidate is not a
    # gate at all. Set from the measured split: genuine gates sit near 0.6,
    # the misread door that prompted this scored 0.28.
    # How far, in pixels, a label's sides may sit from the image's own edges.
    # Measured medians: 1.25px unclipped, 2.75px clipped, 3.25px for the
    # largest gates - all accurate; the spread is measurement difficulty, not
    # label error, so the bar is set in absolute pixels.
    maximum_alignment_px: float = 4.0
    # Longest-over-shortest side of the outer quad. Past this the gate is so
    # nearly edge-on that its opening is not meaningfully observable.
    maximum_aspect: float = 4.0
    # Claimed corners must span at least this share of the gate side they
    # assert, or they are sitting on something other than this gate.
    minimum_corner_spread: float = 0.25
    # Also cap the error as a share of gate size, so a 70px gate cannot pass on
    # an absolute tolerance that would be a twentieth of it.
    relative_alignment: float = 0.045
    # Pixel budget grows with the gate, because a nearer gate's edges are
    # physically wider in the image.
    alignment_size_share: float = 0.008
    # Probes one SIDE must afford before it counts as measured. A corner needs
    # both of its sides measured before the label claims it.
    # Probes one SIDE must afford before it counts as measured. Lowering this
    # mainly helps close gates, whose sides are largely off the frame: 5 leaves
    # 46% of them with enough points for PnP, 3 gives 48%, 2 gives 54%. Two is
    # thin evidence for a line, and a corner still needs BOTH its sides to
    # pass, so 3 is the compromise.
    minimum_side_stations: int = 3
    # Corners the image vouches for, below which there is no usable label.
    minimum_confirmed_keypoints: int = 4

    # Share of the opening still showing gate-orange, above which it is not an
    # opening but another piece of gate.
    # Set from the measured split of the rim band: a real gate with another
    # gate seen through it comes in at 0.35, while the signage boards and bar
    # fragments this is for run 0.44 to 0.66.
    maximum_opening_orange: float = 0.42

    appearance_threshold: float = 0.40
    # Below this much of the face on screen, appearance cannot be judged.
    appearance_min_coverage: float = 0.55
    appearance_gate: bool = True
    appearance_teacher_support: float = 0.88
    appearance_teacher_side: float = 220.0
    # Perpendicular probes a ring must afford before its score means anything.
    minimum_stations: int = 8
    # Head start for a proposal whose two rings were measured independently and
    # agreed, over one ring extrapolated into the other.
    cross_check_bonus: float = 0.12

    accept_residual: float = 0.035
    review_residual: float = 0.130
    subpix_window: int = 5
    edge_refine: bool = True
    edge_refine_radius: float = 14.0
    # Swept on 403 frames. Mild proximity weighting beats taking the
    # strongest step outright: it holds the tail (a gate seen through the
    # opening offers a stronger edge than the opening's own) without
    # dragging the median back toward the threshold boundary.
    # Two passes: the first travels to the real edge, the second polishes
    # in a tighter band. A third adds nothing.
    region_aperture: bool = True
    mask_opening: bool = True
    rescue_unmeasured: bool = True
    edge_refine_passes: int = 2
    edge_proximity_frac: float = 1.0
    edge_size_frac: float = 0.0


@dataclass
class GateInstance:
    outer: np.ndarray
    inner: np.ndarray
    residual: float          # normalised opening-prediction miss, inf if unchecked
    outer_side_px: float
    clipped: bool
    support: float = 0.0          # weakest measurable ring
    projected_support: float = -1.0  # the ring that was NOT measured; -1 unmeasurable
    appearance: float = -2.0         # correlation with the learned gate face
    alignment: float = math.inf      # median px from the label's sides to real edges
    suppress: str = ""               # a ring whose points are not claimed
    verified: np.ndarray | None = None  # per-keypoint: two measured sides agree
    anchor: str = "both"     # which ring was actually observed
    reason: str = ""
    flags: list[str] = field(default_factory=list)   # force a human look
    notes: list[str] = field(default_factory=list)   # describe, do not gate

    @property
    def points(self) -> np.ndarray:
        return np.vstack([self.outer, self.inner])

    @property
    def verdict(self) -> str:
        if self.reason:
            return "rejected"
        return "auto" if not self.flags else "review"

    @property
    def tags(self) -> str:
        return "+".join(self.flags + [f"({n})" for n in self.notes])


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points TL, TR, BR, BL - the cyclic rule vision/ already uses."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    cyclic = pts[np.argsort(angles, kind="stable")]
    start = int(np.argmin(cyclic.sum(axis=1)))
    return np.roll(cyclic, -start, axis=0).astype(np.float32)


def orange_mask(image: np.ndarray, config: AutolabelConfig) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lower, upper in config.hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    if config.close_kernel:
        kernel = np.ones((config.close_kernel, config.close_kernel), np.uint8)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel, iterations=config.close_iterations
        )
    if config.open_kernel:
        kernel = np.ones((config.open_kernel, config.open_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _edge_on_border(a: np.ndarray, b: np.ndarray, shape, margin: int = 5) -> bool:
    """True when a hull edge hugs an image edge - that is the crop, not the gate."""
    height, width = shape[:2]
    if a[0] <= margin and b[0] <= margin:
        return True
    if a[1] <= margin and b[1] <= margin:
        return True
    if a[0] >= width - 1 - margin and b[0] >= width - 1 - margin:
        return True
    if a[1] >= height - 1 - margin and b[1] >= height - 1 - margin:
        return True
    return False


def _fit_line(points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    vx, vy, x0, y0 = cv2.fitLine(
        np.asarray(points, np.float32), cv2.DIST_L2, 0, 0.01, 0.01
    ).ravel()
    return np.array([x0, y0], float), np.array([vx, vy], float)


def _intersect(first, second) -> np.ndarray | None:
    (origin_a, direction_a), (origin_b, direction_b) = first, second
    cross = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    if abs(cross) < 1e-9:
        return None
    step = (
        (origin_b[0] - origin_a[0]) * direction_b[1]
        - (origin_b[1] - origin_a[1]) * direction_b[0]
    ) / cross
    return origin_a + step * direction_a


def quad_from_hull(
    contour: np.ndarray, shape, angle_tol: float = 28.0, min_frac: float = 0.07,
    clipped: bool = False,
) -> np.ndarray | None:
    """Recover a quad from its sides, so a clipped gate still yields corners.

    The convex hull of the orange region is the outer square and nothing else -
    the opening and the hexagon dots are interior, so the hull steps over them.
    Hull runs along an image edge are where the frame cut the gate; the rest are
    real sides. Fit each side, intersect consecutive pairs, and corners beyond
    the image come back with them. Needs all four sides to show something.
    """
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(float)
    if len(hull) < 3:
        return None
    scale = math.sqrt(max(cv2.contourArea(contour), 1.0))

    edges: list[tuple[list[np.ndarray], float]] = []
    for index in range(len(hull)):
        a, b = hull[index], hull[(index + 1) % len(hull)]
        if np.linalg.norm(b - a) < min_frac * scale:
            continue
        if _edge_on_border(a, b, shape):
            continue
        edges.append(([a, b], math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0))
    if len(edges) < 4:
        return None

    # One side can arrive as several hull edges; merge the ones pointing the same way.
    groups: list[list] = [[edges[0]]]
    for edge in edges[1:]:
        delta = abs(edge[1] - groups[-1][-1][1])
        if min(delta, 180.0 - delta) < angle_tol:
            groups[-1].append(edge)
        else:
            groups.append([edge])
    if len(groups) > 1:
        delta = abs(groups[0][0][1] - groups[-1][-1][1])
        if min(delta, 180.0 - delta) < angle_tol:
            groups[0] = groups[-1] + groups[0]
            groups.pop()
    if len(groups) != 4:
        return None

    lines = [_fit_line([p for edge in group for p in edge[0]]) for group in groups]
    corners = []
    for index in range(4):
        point = _intersect(lines[index], lines[(index + 1) % 4])
        if point is None:
            return None
        corners.append(point)
    quad = order_corners(np.array(corners, np.float32))
    # A gate ring's hull IS its outer square, so a correct quad and the hull
    # should have nearly the same area. When two gates' orange fuses into one
    # contour the fitted sides belong to different gates, and intersecting them
    # throws a corner off toward the vanishing point - which shows up here as a
    # quad far larger than the hull it came from. A clipped gate legitimately
    # extrapolates well beyond its visible sliver, so it gets more room.
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    quad_area = cv2.contourArea(quad)
    if hull_area <= 0 or quad_area < 0.55 * hull_area:
        return None
    if quad_area > (2.6 if clipped else 1.7) * hull_area:
        return None
    return quad


def fit_quadrilateral(contour: np.ndarray, allow_box: bool = True) -> np.ndarray | None:
    """Four convex corners, or None. Tries tightening/loosening the tolerance."""
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None
    for fraction in (0.02, 0.01, 0.03, 0.045, 0.06):
        approx = cv2.approxPolyDP(contour, fraction * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    if not allow_box:
        return None
    # A corner clipped by the frame edge turns the quad into a polygon; the
    # minimum-area rectangle still recovers a usable four-point hypothesis.
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area > 0 and cv2.contourArea(box.astype(np.float32)) / hull_area < 1.45:
        return box.astype(np.float32)
    return None


def refine_subpixel(gray: np.ndarray, points: np.ndarray, window: int) -> np.ndarray:
    """Sub-pixel refinement for the corners that have pixels to refine against.

    A gate wider than the frame puts real corners outside it, and minAreaRect
    can propose one just past the edge. cornerSubPix raises on those, so they
    keep their fitted position - it is projected geometry, not an observation,
    and there is nothing in the image to snap it to.
    """
    height, width = gray.shape[:2]
    margin = window + 2
    refined = points.astype(np.float32).copy()
    inside = (
        (points[:, 0] >= margin)
        & (points[:, 0] < width - margin)
        & (points[:, 1] >= margin)
        & (points[:, 1] < height - margin)
    )
    if not inside.any():
        return refined

    corners = points[inside].reshape(-1, 1, 2).astype(np.float32).copy()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    cv2.cornerSubPix(gray, corners, (window, window), (-1, -1), criteria)
    moved = corners.reshape(-1, 2)
    # cornerSubPix can bolt onto a nearby texture edge - the hexagon pattern and
    # sponsor logos sit right beside the real corners. Keep it on a short leash.
    drift = np.linalg.norm(moved - points[inside], axis=1)
    moved[drift > window * 1.5] = points[inside][drift > window * 1.5]
    refined[inside] = moved
    return refined


def orange_field(image: np.ndarray) -> np.ndarray:
    """Continuous 'how orange is this pixel', for sub-pixel edge work.

    The binary mask says where the threshold fell, which is a property of the
    threshold as much as of the gate. This keeps the underlying ramp, so an
    edge can be located at the real transition instead of wherever inRange
    happened to cut. LAB's a* axis is red-versus-green, which is exactly the
    contrast between safety orange and a grey hangar.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return lab[:, :, 1].astype(np.float32)


def _sample_bilinear(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.remap(
        field, points[:, 0].astype(np.float32).reshape(-1, 1),
        points[:, 1].astype(np.float32).reshape(-1, 1),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    ).ravel()


def refine_edges_subpixel(
    field: np.ndarray, quad: np.ndarray, radius: float = 7.0, step: float = 0.25,
    proximity_frac: float = 0.5, size_frac: float = 0.06,
) -> np.ndarray | None:
    """Re-fit each side to the image's own colour step, then re-intersect.

    Walks each side, and at every station looks along the perpendicular for the
    steepest change in orangeness, to sub-pixel precision. Those stations are
    what the side really is; the corners follow from intersecting the four
    re-fitted sides. Corner neighbourhoods are skipped because both sides are
    changing there and the profile is not a clean step.
    """
    height, width = field.shape[:2]
    # Search no further than the gate can afford. On a small gate a wide search
    # reaches across the opening and can settle on the far side of it.
    if size_frac > 0:
        radius = float(np.clip(size_frac * mean_side_length(quad), 3.0, radius))
    offsets = np.arange(-radius, radius + step, step, dtype=np.float32)
    # Another gate seen through this one's opening puts a strong orange edge
    # inside the search window, and it can be stronger than the edge we want.
    # Preferring the nearest credible step over the strongest one keeps the
    # refinement a refinement instead of a jump to a different gate.
    proximity = (
        np.exp(-0.5 * (offsets / max(radius * proximity_frac, 1e-6)) ** 2)
        if proximity_frac > 0 else np.ones_like(offsets)
    )
    lines = []
    for index in range(4):
        start, end = quad[index], quad[(index + 1) % 4]
        length = float(np.linalg.norm(end - start))
        if length < 12.0:
            return None
        direction = (end - start) / length
        normal = np.array([-direction[1], direction[0]], np.float32)
        stations = np.linspace(0.18, 0.82, int(np.clip(length / 6.0, 9, 80)))

        found = []
        for t in stations:
            base = start + t * length * direction
            samples = base[None, :] + offsets[:, None] * normal[None, :]
            if (samples[:, 0] < 1).any() or (samples[:, 0] > width - 2).any():
                continue
            if (samples[:, 1] < 1).any() or (samples[:, 1] > height - 2).any():
                continue
            profile = _sample_bilinear(field, samples)
            gradient = np.abs(np.gradient(profile))
            peak = int(np.argmax(gradient * proximity))
            if peak == 0 or peak == len(gradient) - 1:
                continue
            # Parabola through the peak and its neighbours -> sub-sample offset.
            a, b, c = gradient[peak - 1], gradient[peak], gradient[peak + 1]
            denominator = a - 2.0 * b + c
            shift = 0.0 if abs(denominator) < 1e-9 else 0.5 * (a - c) / denominator
            if abs(shift) > 1.0:
                continue
            found.append(base + (offsets[peak] + shift * step) * normal)

        if len(found) < 6:
            return None
        # Huber keeps a glare patch or an occluder from dragging the whole side.
        vx, vy, x0, y0 = cv2.fitLine(
            np.asarray(found, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
        ).ravel()
        lines.append((np.array([x0, y0], float), np.array([vx, vy], float)))

    corners = []
    for index in range(4):
        point = _intersect(lines[index], lines[(index + 1) % 4])
        if point is None:
            return None
        corners.append(point)
    refined = order_corners(np.array(corners, np.float32))
    # A refinement that moves a corner further than this is not a refinement.
    if (np.linalg.norm(refined - quad, axis=1) > 0.25 * mean_side_length(quad)).any():
        return None
    return refined


def aperture_from_region(
    mask: np.ndarray, outer_quad: np.ndarray, shape, config: "AutolabelConfig"
) -> np.ndarray | None:
    """Find the opening as a region rather than as a closed contour hole.

    When the frame cuts a gate, the opening stops being a hole - it spills over
    the image edge and merges with the outside, so contour hierarchy loses it
    and the gate is left anchored on one ring. But the opening is still plainly
    there: inside the outer square, everything that is not orange. Taking the
    largest such region and fitting its sides recovers the ring that the
    hierarchy dropped, which turns a single-anchor guess into a cross-checked
    measurement.
    """
    height, width = shape[:2]
    inside = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(inside, np.round(outer_quad).astype(np.int32), 255)
    inside_area = float(np.count_nonzero(inside))
    if inside_area < 400:
        return None

    region = cv2.bitwise_and(inside, cv2.bitwise_not(mask))
    # Drop the hexagon dots, the sponsor text and the threshold's ragged fringe.
    kernel = np.ones((5, 5), np.uint8)
    region = cv2.morphologyEx(region, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    # Against the part of the square actually on screen, not the whole square.
    if not 0.08 * inside_area < area < 0.75 * inside_area:
        return None
    for candidate in (
        quad_from_hull(best, shape, clipped=True),
        fit_quadrilateral(best),
    ):
        if candidate is not None and _plausible_quad(candidate, width, height):
            return order_corners(candidate)
    return None


TEMPLATE_SIZE = 192
TEMPLATE_MARGIN = 0.60
_TEMPLATE_CANONICAL = (
    CANONICAL_OUTER * TEMPLATE_SIZE * TEMPLATE_MARGIN + TEMPLATE_SIZE / 2
).astype(np.float32)


@dataclass
class GateAppearance:
    """What a gate face looks like, learned from this capture's own labels."""

    mean: np.ndarray
    weight: np.ndarray

    def score(self, gray: np.ndarray, outer: np.ndarray) -> tuple[float, float]:
        """Correlation with the learned face, and how much of it was in frame."""
        matrix, _ = cv2.findHomography(
            outer.astype(np.float32), _TEMPLATE_CANONICAL, 0
        )
        if matrix is None:
            return -1.0, 0.0
        inside = cv2.warpPerspective(
            np.ones(gray.shape[:2], np.float32), matrix,
            (TEMPLATE_SIZE, TEMPLATE_SIZE), flags=cv2.INTER_NEAREST,
        )
        coverage = float(inside.mean())
        patch = cv2.warpPerspective(
            gray, matrix, (TEMPLATE_SIZE, TEMPLATE_SIZE), flags=cv2.INTER_LINEAR
        )
        if patch.std() < 1e-3:
            return -1.0, coverage
        patch = (patch - patch.mean()) / (patch.std() + 1e-6)
        # Corner ids are assigned by image position, so a gate seen with camera
        # roll has its face rectified a quarter turn from the template even
        # when the label is perfect. Comparing against all four assignments
        # keeps the check about whether this is a gate, not how it was rolled.
        best = -1.0
        for turn in range(4):
            turned = np.rot90(patch, turn)
            best = max(best, float((turned * self.mean * self.weight).mean()))
        return best, coverage


def learn_appearance(samples: list[np.ndarray]) -> GateAppearance | None:
    """Median of many rectified gate faces, plus per-pixel agreement weights.

    The median keeps an occluder or a passing person from dragging the result,
    and weighting by agreement lets the parts that are always the same - the
    wordmark, the hexagon columns, the sponsor row - carry the comparison,
    while whatever happens to show through the opening carries little.
    """
    if len(samples) < 12:
        return None
    stack = np.array(samples)
    mean = np.median(stack, axis=0)
    mean = (mean - mean.mean()) / (mean.std() + 1e-6)
    weight = 1.0 / (1.0 + stack.std(axis=0))
    weight /= weight.mean()
    return GateAppearance(mean.astype(np.float32), weight.astype(np.float32))


def rectified_face(gray: np.ndarray, outer: np.ndarray) -> np.ndarray | None:
    matrix, _ = cv2.findHomography(outer.astype(np.float32), _TEMPLATE_CANONICAL, 0)
    if matrix is None:
        return None
    patch = cv2.warpPerspective(
        gray, matrix, (TEMPLATE_SIZE, TEMPLATE_SIZE), flags=cv2.INTER_LINEAR
    )
    if patch.std() < 1e-3:
        return None
    return ((patch - patch.mean()) / (patch.std() + 1e-6)).astype(np.float32)


def _group_sides(
    segments: list[tuple[np.ndarray, np.ndarray, float]], angle_tol: float = 26.0
) -> list[list[tuple[np.ndarray, np.ndarray, float]]]:
    """Merge consecutive segments that point the same way into one side."""
    if not segments:
        return []
    groups = [[segments[0]]]
    for segment in segments[1:]:
        delta = abs(segment[2] - groups[-1][-1][2])
        if min(delta, 180.0 - delta) < angle_tol:
            groups[-1].append(segment)
        else:
            groups.append([segment])
    if len(groups) > 1:
        delta = abs(groups[0][0][2] - groups[-1][-1][2])
        if min(delta, 180.0 - delta) < angle_tol:
            groups[0] = groups[-1] + groups[0]
            groups.pop()
    return groups


def aperture_sides_from_concavity(
    contour: np.ndarray, shape, minimum_depth: float = 0.035,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Recover the opening's sides from the gate outline's concave parts.

    When the frame cuts a gate, the opening stops being a hole: it spills over
    the image edge, merges with the outside, and contour hierarchy loses it.
    But it has not gone anywhere. The opening's rim is still traced by the same
    outline, as the stretch that dips inside the convex hull - the hull spans
    the mouth while the contour walks around the rim.

    So take the points lying well inside the hull, keep the straight runs among
    them, and fit a line to each. That yields however many sides of the opening
    are actually visible: often two or three, which is not enough to trace the
    ring by itself but is plenty as evidence for a joint solve.
    """
    height, width = shape[:2]
    area = cv2.contourArea(contour)
    if area <= 0:
        return []
    scale = math.sqrt(area)
    perimeter = cv2.arcLength(contour, True)
    poly = cv2.approxPolyDP(contour, 0.004 * perimeter, True).reshape(-1, 2).astype(np.float32)
    if len(poly) < 6:
        return []
    hull = cv2.convexHull(poly.reshape(-1, 1, 2)).astype(np.float32)

    depth = np.array([
        -cv2.pointPolygonTest(hull, (float(point[0]), float(point[1])), True)
        for point in poly
    ])
    # Negative means inside the hull; we want the points that dip well inside.
    inner = depth < -minimum_depth * scale

    segments: list[tuple[np.ndarray, np.ndarray, float]] = []
    for index in range(len(poly)):
        nxt = (index + 1) % len(poly)
        if not (inner[index] and inner[nxt]):
            continue
        a, b = poly[index], poly[nxt]
        if np.linalg.norm(b - a) < 0.05 * scale:
            continue
        if _edge_on_border(a, b, shape):
            continue
        segments.append((a, b, math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0))

    sides = []
    for group in _group_sides(segments):
        span = sum(float(np.linalg.norm(b - a)) for a, b, _ in group)
        if span < 0.10 * scale:
            continue
        sides.append(_fit_line([p for a, b, _ in group for p in (a, b)]))
    return sides


def side_alignments(
    field: np.ndarray, quad: np.ndarray, radius: float = 6.0, step: float = 0.25,
) -> list[tuple[float, int]]:
    """Per side: how far the image's edge sits from it, and how many probes."""
    height, width = field.shape[:2]
    offsets = np.arange(-radius, radius + step, step, dtype=np.float32)
    out: list[tuple[float, int]] = []
    for index in range(4):
        start, end = quad[index], quad[(index + 1) % 4]
        length = float(np.linalg.norm(end - start))
        if length < 10.0:
            out.append((math.inf, 0))
            continue
        direction = (end - start) / length
        normal = np.array([-direction[1], direction[0]], np.float32)
        misses: list[float] = []
        for t in np.linspace(SIDE_SPAN_LO, SIDE_SPAN_HI,
                             int(np.clip(length / 10.0, 5, 40))):
            base = start + t * length * direction
            samples = base[None, :] + offsets[:, None] * normal[None, :]
            if (samples[:, 0] < 1).any() or (samples[:, 0] > width - 2).any():
                continue
            if (samples[:, 1] < 1).any() or (samples[:, 1] > height - 2).any():
                continue
            profile = _sample_bilinear(field, samples)
            if float(profile.max() - profile.min()) < 12.0:
                continue
            gradient = np.abs(np.gradient(profile))
            misses.append(abs(float(offsets[int(np.argmax(gradient))])))
        if not misses:
            out.append((math.inf, 0))
        else:
            out.append((float(np.median(misses)), len(misses)))
    return out


def verified_keypoints(
    field: np.ndarray, points: np.ndarray, allowed: float,
    minimum_stations: int,
) -> np.ndarray:
    """Which of the eight keypoints two measured sides actually vouch for.

    Checking a whole ring at once lets a corner ride in on its neighbours. A
    ring can have three sides sitting perfectly on real edges and a fourth that
    was never measured at all - off the frame, or facing nothing with any
    contrast - and the two corners on that fourth side are then pure
    extrapolation while the ring's average still looks excellent.

    A corner is where two sides meet, so it is worth exactly as much as the
    weaker of them. Corners whose sides were not both measured are not claimed.
    """
    verified = np.zeros(8, bool)
    for ring_start in (0, 4):
        ring = points[ring_start:ring_start + 4]
        measured = [
            count >= minimum_stations and offset <= allowed
            for offset, count in side_alignments(field, ring)
        ]
        for corner in range(4):
            # Corner c is the meeting of side c-1 (into it) and side c (out).
            verified[ring_start + corner] = (
                measured[(corner - 1) % 4] and measured[corner]
            )
    return verified


def opening_is_see_through(
    image: np.ndarray, mask: np.ndarray, inner: np.ndarray,
) -> tuple[float, float]:
    """Does the opening show the room behind, or is it printed artwork?

    The course carries flat AI-GP signage boards with the same wordmark,
    hexagon columns and sponsor row as the gates - so colour, shape and even
    the learned gate appearance all say "gate". The one thing a board cannot
    fake is the hole. Look through a real opening and the hangar is there:
    floor, far wall, lights, other gates, all varying. A board's "opening" is
    a flat panel of print.

    Returns the share of the opening that is still gate-orange, and how much
    the rest of it varies.
    """
    height, width = image.shape[:2]
    region = np.zeros((height, width), np.uint8)
    quad = np.round(inner).astype(np.int32)
    cv2.fillConvexPoly(region, quad, 255)
    # Pull in from the rim so the edge itself is not what we measure.
    region = cv2.erode(region, np.ones((9, 9), np.uint8), iterations=2)
    pixels = int(np.count_nonzero(region))
    if pixels < 120:
        return 0.0, math.inf

    # Measure just inside the rim rather than the whole opening. A gate
    # standing behind this one is seen through the middle, with the room
    # visible around it - so the rim band stays dark. A piece of bar has gate
    # surface continuous right up to the edge. Sampling the whole opening
    # confuses the two and throws away perfectly good gate-behind-gate shots.
    inner_core = cv2.erode(region, np.ones((9, 9), np.uint8),
                           iterations=max(int(math.sqrt(pixels) / 14), 1))
    band = cv2.subtract(region, inner_core)
    band_pixels = int(np.count_nonzero(band))
    if band_pixels < 60:
        band, band_pixels = region, pixels
    orange = float(np.count_nonzero(cv2.bitwise_and(band, mask))) / band_pixels
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inside = grey[region > 0].astype(np.float32)
    return orange, float(inside.std())


def quad_from_sides(sides: list[tuple[np.ndarray, np.ndarray]], shape) -> np.ndarray | None:
    """Four fitted sides, in order around the ring, become four corners."""
    if len(sides) != 4:
        return None
    height, width = shape[:2]
    corners = []
    for index in range(4):
        point = _intersect(sides[index], sides[(index + 1) % 4])
        if point is None:
            return None
        corners.append(point)
    quad = order_corners(np.array(corners, np.float32))
    if not _plausible_quad(quad, width, height):
        return None
    return quad


def _outward_orange(mask: np.ndarray, quad: np.ndarray, step: float = 9.0) -> list:
    """Per side: the share of in-frame stations with orange just beyond it."""
    height, width = mask.shape[:2]
    centre = quad.mean(axis=0)
    shares = []
    for index in range(4):
        a, b = quad[index], quad[(index + 1) % 4]
        length = float(np.linalg.norm(b - a))
        if length < 12:
            shares.append(None)
            continue
        direction = (b - a) / length
        normal = np.array([-direction[1], direction[0]], np.float32)
        if np.dot(normal, (a + b) / 2 - centre) < 0:
            normal = -normal                       # make it point outward
        hit = total = 0
        for t in np.linspace(0.12, 0.88, 25):
            point = a + t * length * direction + normal * step
            x, y = int(round(point[0])), int(round(point[1]))
            if not (0 <= x < width and 0 <= y < height):
                continue
            total += 1
            hit += mask[y, x] > 0
        shares.append(hit / total if total >= 5 else None)
    return shares


def opening_from_mask(
    mask: np.ndarray, shape, minimum_area: int = 4000,
    minimum_outward: float = 0.80,
) -> np.ndarray | None:
    """Find the opening from the mask alone, without needing the outer square.

    On a close gate the outer square is the least reliable thing in the frame,
    because most of it is outside the frame - and every other route to the
    opening derives it from that square, so a bad outer quad yields a
    meaningless opening. The opening itself is plainly present though: a region
    of not-orange that the gate encloses.

    Counting boundary pixels cannot separate it from the room beyond a clipped
    gate, since both are bounded by the same silhouette. Direction can. Step
    outward from a side of the opening and you land on the gate; step outward
    from the room and you land on more room.
    """
    height, width = shape[:2]
    inverted = cv2.bitwise_not(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, 8)
    best = None
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < minimum_area:
            continue
        component = (labels == index).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        quad = quad_from_hull(contour, shape, clipped=True)
        if quad is None:
            quad = fit_quadrilateral(contour)
        if quad is None or not _plausible_quad(quad, width, height):
            continue
        shares = [v for v in _outward_orange(mask, quad) if v is not None]
        if len(shares) < 2 or float(np.mean(shares)) < minimum_outward:
            continue
        if best is None or mean_side_length(quad) > mean_side_length(best):
            best = order_corners(quad)
    return best


def edge_support(
    field: np.ndarray, quad: np.ndarray, tolerance: float = 2.5,
    radius: float = 6.0, step: float = 0.25,
) -> tuple[float, int]:
    """Does the image actually have an edge where this quad says it does?

    Independent of how the quad was arrived at. Walks each side and asks, at
    every station, whether the strongest colour step within a short look really
    sits on the line. A quad traced from a contour will score well because it
    came from that edge; the value is for a quad that was *predicted* - a ring
    projected through a homography rather than measured - where a high score is
    genuine confirmation from pixels the prediction never saw.

    Returns the supported fraction and how many stations could be sampled at
    all. A ring the frame has mostly cut away yields few stations, and few
    stations is weak evidence - but it is not the same as evidence against.
    """
    height, width = field.shape[:2]
    offsets = np.arange(-radius, radius + step, step, dtype=np.float32)
    supported = total = 0
    for index in range(4):
        start, end = quad[index], quad[(index + 1) % 4]
        length = float(np.linalg.norm(end - start))
        if length < 10.0:
            continue
        direction = (end - start) / length
        normal = np.array([-direction[1], direction[0]], np.float32)
        side_hits = side_total = 0
        for t in np.linspace(SIDE_SPAN_LO, SIDE_SPAN_HI,
                             int(np.clip(length / 10.0, 5, 40))):
            base = start + t * length * direction
            samples = base[None, :] + offsets[:, None] * normal[None, :]
            if (samples[:, 0] < 1).any() or (samples[:, 0] > width - 2).any():
                continue
            if (samples[:, 1] < 1).any() or (samples[:, 1] > height - 2).any():
                continue
            profile = _sample_bilinear(field, samples)
            spread = float(profile.max() - profile.min())
            if spread < 12.0:
                # Flat here: no edge of any kind, so this station cannot vote.
                continue
            gradient = np.abs(np.gradient(profile))
            side_total += 1
            if abs(float(offsets[int(np.argmax(gradient))])) <= tolerance:
                side_hits += 1
        # Even two usable stations are evidence; requiring a whole clean side
        # silently scored every heavily cropped gate as zero, which is not the
        # same as unsupported.
        if side_total >= 2:
            supported += side_hits
            total += side_total
    if total == 0:
        return 0.0, 0
    return supported / total, total


def edge_alignment(
    field: np.ndarray, quad: np.ndarray, radius: float = 6.0, step: float = 0.25,
) -> tuple[float, int]:
    """Typical distance, in pixels, from the label's sides to the image's edges.

    A hit-rate punishes a label for every station that disagrees, but stations
    disagree for reasons that are not the label's fault: the frame border
    truncating the probe, something passing in front, motion blur, or the
    260mm tunnel wall showing a second edge beside the real one. A gate the
    frame has cut has few stations left, so each such failure costs it
    disproportionately - which made the score systematically worse for exactly
    the close gates that matter most.

    The median offset does not care about a minority of distracted stations. It
    asks the question that actually matters: when the image does show an edge
    near this side, how far is it? Returns (median offset in pixels, stations).
    """
    height, width = field.shape[:2]
    offsets = np.arange(-radius, radius + step, step, dtype=np.float32)
    misses: list[float] = []
    for index in range(4):
        start, end = quad[index], quad[(index + 1) % 4]
        length = float(np.linalg.norm(end - start))
        if length < 10.0:
            continue
        direction = (end - start) / length
        normal = np.array([-direction[1], direction[0]], np.float32)
        for t in np.linspace(SIDE_SPAN_LO, SIDE_SPAN_HI,
                             int(np.clip(length / 10.0, 5, 40))):
            base = start + t * length * direction
            samples = base[None, :] + offsets[:, None] * normal[None, :]
            if (samples[:, 0] < 1).any() or (samples[:, 0] > width - 2).any():
                continue
            if (samples[:, 1] < 1).any() or (samples[:, 1] > height - 2).any():
                continue
            profile = _sample_bilinear(field, samples)
            if float(profile.max() - profile.min()) < 12.0:
                continue
            gradient = np.abs(np.gradient(profile))
            misses.append(abs(float(offsets[int(np.argmax(gradient))])))
    if len(misses) < 4:
        return math.inf, len(misses)
    return float(np.median(misses)), len(misses)


def refine_pose_jointly(
    field: np.ndarray, points: np.ndarray, radius: float = 10.0, step: float = 0.25,
    passes: int = 3,
) -> np.ndarray | None:
    """Re-fit all eight points as one gate, using every edge still on screen.

    Fitting the rings separately wastes the fact that they are one rigid shape.
    A gate the frame has cut may show only two sides of its outer square and
    two of its opening - not four of either, so neither ring can be traced on
    its own - yet those four sides together still pin down the single
    homography that maps the canonical face onto the image.

    So: walk every one of the eight sides, keep the stations where the image
    really does step, and solve one homography against all of them at once.
    Evidence from the opening then constrains the outer corners and vice versa,
    which is exactly what the separate fits could not do.

    Use it only where the normal path has failed. Applied to every label it
    makes things worse: it lifts single-anchor labels (outer .51->.54, inner
    .55->.60) but degrades two-ring ones (.88->.72), because when both rings
    were traced the per-ring sub-pixel fit and the least-squares snap are
    already finer than this solve. Measured against synthetic crops with known
    answers it came out neutral to slightly worse at every reach band - which
    is why this is a rescue for labels that have no fit at all, not a general
    refinement.
    """
    height, width = field.shape[:2]
    offsets = np.arange(-radius, radius + step, step, dtype=np.float32)
    canonical = np.vstack([CANONICAL_OUTER, CANONICAL_INNER]).astype(np.float32)
    current = points.astype(np.float32).copy()

    for attempt in range(passes):
        matrix, _ = cv2.findHomography(canonical, current, 0)
        if matrix is None:
            return None
        source: list[np.ndarray] = []
        target: list[np.ndarray] = []
        for ring_start, ring in ((0, CANONICAL_OUTER), (4, CANONICAL_INNER)):
            for index in range(4):
                a_canonical, b_canonical = ring[index], ring[(index + 1) % 4]
                a_image, b_image = current[ring_start + index], \
                    current[ring_start + (index + 1) % 4]
                length = float(np.linalg.norm(b_image - a_image))
                if length < 14.0:
                    continue
                direction = (b_image - a_image) / length
                normal = np.array([-direction[1], direction[0]], np.float32)
                for t in np.linspace(0.15, 0.85, int(np.clip(length / 8.0, 6, 40))):
                    base = a_image + t * length * direction
                    samples = base[None, :] + offsets[:, None] * normal[None, :]
                    if (samples[:, 0] < 1).any() or (samples[:, 0] > width - 2).any():
                        continue
                    if (samples[:, 1] < 1).any() or (samples[:, 1] > height - 2).any():
                        continue
                    profile = _sample_bilinear(field, samples)
                    if float(profile.max() - profile.min()) < 14.0:
                        continue
                    gradient = np.abs(np.gradient(profile))
                    peak = int(np.argmax(gradient))
                    if peak == 0 or peak == len(gradient) - 1:
                        continue
                    left, mid, right = (gradient[peak - 1], gradient[peak],
                                        gradient[peak + 1])
                    denominator = left - 2.0 * mid + right
                    shift = (0.0 if abs(denominator) < 1e-9
                             else 0.5 * (left - right) / denominator)
                    if abs(shift) > 1.0:
                        continue
                    found = base + (offsets[peak] + shift * step) * normal
                    source.append(a_canonical + t * (b_canonical - a_canonical))
                    target.append(found)

        if len(source) < 24:
            return None
        updated, inliers = cv2.findHomography(
            np.array(source, np.float32), np.array(target, np.float32),
            cv2.RANSAC, 2.5,
        )
        if updated is None or inliers is None or int(inliers.sum()) < 20:
            return None
        proposal = cv2.perspectiveTransform(
            canonical.reshape(-1, 1, 2), updated
        ).reshape(8, 2)
        moved = float(np.linalg.norm(proposal - current, axis=1).max())
        current = proposal.astype(np.float32)
        if moved < 0.4:
            break

    # A refinement that relocates the gate is not a refinement.
    if (np.linalg.norm(current - points, axis=1)
            > 0.30 * mean_side_length(points[:4])).any():
        return None
    return current


def mean_side_length(quad: np.ndarray) -> float:
    return float(
        np.mean([np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)])
    )


def opening_residual(outer: np.ndarray, inner: np.ndarray) -> tuple[float, np.ndarray]:
    """How far the detected opening sits from where the front face demands it.

    Fits the canonical outer square to the detected outer quad, then projects
    the canonical opening through that same homography. Perspective-correct, so
    unlike a raw side-length ratio it stays honest on obliquely viewed gates.
    Normalised by outer side length, so it is scale-invariant too.
    """
    matrix, _ = cv2.findHomography(CANONICAL_OUTER, outer, 0)
    if matrix is None:
        return math.inf, inner
    predicted = cv2.perspectiveTransform(
        CANONICAL_INNER.reshape(-1, 1, 2), matrix
    ).reshape(4, 2)
    side = mean_side_length(outer)
    if side <= 0:
        return math.inf, predicted
    miss = float(np.mean(np.linalg.norm(predicted - inner, axis=1)))
    return miss / side, predicted


def normalise_ring_order(points: np.ndarray) -> np.ndarray:
    """Put both rings in TL, TR, BR, BL without breaking their pairing.

    A homography carrying a reflection, or a gate rolled far enough that its
    physical top-left is no longer the image's top-left, leaves the canonical
    index order disagreeing with the convention the rest of the repo assumes.
    Re-ordering each ring on its own would fix the convention and silently
    break correspondence - outer id 0 and inner id 4 have to stay the same
    physical corner - so the outer ring decides the permutation and the inner
    ring follows it.
    """
    outer, inner = points[:4], points[4:]
    ordered = order_corners(outer)
    permutation = []
    for target in ordered:
        distances = np.linalg.norm(outer - target, axis=1)
        permutation.append(int(np.argmin(distances)))
    if sorted(permutation) != [0, 1, 2, 3]:
        return points  # ambiguous match; leave it rather than scramble it
    return np.vstack([outer[permutation], inner[permutation]])


def _rectify(points: np.ndarray) -> np.ndarray:
    """Snap eight detected points onto the gate shape they must actually form."""
    canonical = np.vstack([CANONICAL_OUTER, CANONICAL_INNER]).astype(np.float32)
    matrix, _ = cv2.findHomography(canonical, points.astype(np.float32), 0)
    if matrix is None:
        return points
    return cv2.perspectiveTransform(
        canonical.reshape(-1, 1, 2), matrix
    ).reshape(8, 2)


def _project_eight(anchor_quad: np.ndarray, canonical: np.ndarray):
    """Homography from one observed ring to the canonical face, then all eight."""
    matrix, _ = cv2.findHomography(canonical, anchor_quad, 0)
    if matrix is None:
        return None
    both = np.vstack([CANONICAL_OUTER, CANONICAL_INNER]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(both, matrix).reshape(8, 2)


def gate_candidates(
    image: np.ndarray, config: AutolabelConfig,
    appearance: "GateAppearance | None" = None,
) -> list[GateInstance]:
    """Find gates, preferring rings that are actually observed over projected ones.

    Both rings visible is the good case: they cross-check each other and the
    residual is a real measurement. When the frame cuts the gate, whichever ring
    survives anchors the homography alone - usable, but with nothing to check it
    against, so it always goes to review.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = orange_mask(image, config)
    field = orange_field(image) if config.edge_refine else None
    window = config.subpix_window

    # One pixel of background all round: without it a gate touching the frame
    # edge has its real boundary welded to the image boundary in one run, and
    # the crop becomes indistinguishable from the gate.
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    contours, hierarchy = cv2.findContours(
        padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [contour - 1 for contour in contours]
    if hierarchy is None:
        return []

    instances: list[GateInstance] = []
    hierarchy = hierarchy[0]
    for index, contour in enumerate(contours):
        if hierarchy[index][3] != -1:
            continue  # a hole; it is reached through its parent
        outer_area = cv2.contourArea(contour)
        if outer_area < config.minimum_outer_side_px ** 2 * 0.25:
            continue

        holes = [
            other
            for other in range(len(contours))
            if hierarchy[other][3] == index
            and config.minimum_hole_area_fraction * outer_area
            < cv2.contourArea(contours[other])
            < config.maximum_hole_area_fraction * outer_area
        ]
        touches_edge = _touches_border(contour, width, height)
        if not holes and not touches_edge:
            # A gate wholly inside the frame must show its opening. An orange
            # region that is solid all the way through is a banner or a sign.
            continue

        # Two ways to read the outer square, good at opposite things. Tracing
        # the contour is precise while the whole gate is in frame. Intersecting
        # the sides survives a crop and reaches corners past the frame, but it
        # also picks up the stand and any protrusion. Rather than guess, keep
        # both and let the opening decide which one the gate agrees with.
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        outer_options = []
        for candidate in (
            fit_quadrilateral(contour),
            quad_from_hull(contour, image.shape, clipped=touches_edge),
        ):
            if candidate is None or not _plausible_quad(candidate, width, height):
                continue
            # However the quad was arrived at, a gate ring's hull is its outer
            # square. A quad ballooning past the hull is sides from different
            # gates, or minAreaRect boxing an L-shaped crop - not this gate.
            area = cv2.contourArea(order_corners(candidate))
            if hull_area > 0 and area > (2.6 if touches_edge else 1.7) * hull_area:
                continue
            outer_options.append(
                _best_corners(field, gray, candidate, config)
            )

        inner_options = []
        for other in holes:
            # Hull first here: a gate standing behind this one is seen through
            # the opening, and its orange touches the opening's edge, biting a
            # notch out of the hole contour. The hull spans the notch.
            for candidate in (
                quad_from_hull(contours[other], image.shape),
                fit_quadrilateral(contours[other]),
            ):
                if candidate is None or not _plausible_quad(candidate, width, height):
                    continue
                inner_options.append(
                    _best_corners(field, gray, candidate, config)
                )

        # Gates standing side by side touch, and their orange fuses into a
        # single contour with one aperture each. Keeping only the best pair
        # throws the second gate away, so split on distinct apertures first.
        apertures = _distinct_apertures(inner_options)
        if len(apertures) > 1:
            for aperture in apertures:
                projected = _project_eight(aperture, CANONICAL_INNER)
                if projected is None or not _plausible_quad(projected[:4], width, height):
                    continue
                side = mean_side_length(projected[:4])
                if side < config.minimum_outer_side_px:
                    continue
                shared_support = 0.0
                if field is not None:
                    value, stations = edge_support(field, projected[4:])
                    outer_value, outer_stations = edge_support(field, projected[:4])
                    seen = [v for v, n in ((value, stations),
                                           (outer_value, outer_stations))
                            if n >= config.minimum_stations]
                    shared_support = float(min(seen)) if seen else 0.0
                shared_projected = (
                    float(outer_value)
                    if field is not None and outer_stations >= config.minimum_stations
                    else -1.0
                )
                shared = GateInstance(
                    projected[:4], projected[4:], math.inf, side, True,
                    support=shared_support, projected_support=shared_projected,
                    anchor="inner",
                )
                # Same bar as any other single-anchor label: the image has to
                # confirm the ring that was projected rather than measured.
                if (shared_support >= config.confirm_support
                        and shared_projected >= config.confirm_support):
                    shared.notes.append("inner_only")
                else:
                    shared.flags.append("inner_only")
                shared.notes.append("merged_pair")
                shared.notes.append(f"sup{shared_support:.2f}")
                instances.append(shared)
            continue

        # Straight from the mask, needing no outer square. This is the one that
        # works when the gate is close enough to fly through, where the outer
        # square is mostly off the frame and everything derived from it is a
        # guess.
        if config.mask_opening:
            from_mask = opening_from_mask(mask, image.shape)
            if from_mask is not None:
                inner_options.append(_best_corners(field, gray, from_mask, config))

        # The opening's rim survives in the outline even when it is no longer a
        # hole, as the stretch dipping inside the convex hull. Four such sides
        # give the opening directly, with no dependence on the outer ring -
        # which matters because on a heavily cropped gate the outer ring is the
        # least reliable thing available.
        concave_sides = aperture_sides_from_concavity(contour, image.shape)
        if len(concave_sides) == 4:
            from_rim = quad_from_sides(concave_sides, image.shape)
            if from_rim is not None:
                inner_options.append(_best_corners(field, gray, from_rim, config))

        # The hierarchy misses the opening on clipped gates; recover it from the
        # region inside each outer candidate instead.
        if config.region_aperture and outer_options:
            for outer_candidate in outer_options:
                recovered = aperture_from_region(
                    mask, outer_candidate, image.shape, config
                )
                if recovered is not None:
                    inner_options.append(
                        _best_corners(field, gray, recovered, config)
                    )

        outer_quad, inner_quad, residual = None, None, math.inf
        for outer_candidate in outer_options:
            for inner_candidate in inner_options:
                score, _ = opening_residual(outer_candidate, inner_candidate)
                if score < residual:
                    outer_quad, inner_quad, residual = (
                        outer_candidate, inner_candidate, score
                    )
        if outer_quad is None and outer_options:
            # No opening to arbitrate with. A clipped gate needs the side-fit;
            # an uncropped one is read better by its contour.
            outer_quad = outer_options[-1] if touches_edge else outer_options[0]
        if inner_quad is None and inner_options:
            inner_quad = inner_options[0]

        # Build every complete label this contour could support, then let the
        # image choose. Anchoring first and validating afterwards throws away a
        # real gate whenever the two rings disagree, even when one of them was
        # measured perfectly well - which is the usual case on a gate the frame
        # has cut, where the outer square is a guess and the opening is not.
        proposals: list[tuple[str, np.ndarray, float]] = []
        if outer_quad is not None and inner_quad is not None:
            # Eight correspondences for an eight-DOF homography is one
            # constraint more than the fit needs, so least squares pulls both
            # rings onto a single consistent gate. The pre-fit disagreement is
            # kept as the score; measuring it after would only measure the fit.
            proposals.append(
                ("both", _rectify(np.vstack([outer_quad, inner_quad])), residual)
            )
        for name, quad, canonical in (
            ("inner", inner_quad, CANONICAL_INNER),
            ("outer", outer_quad, CANONICAL_OUTER),
        ):
            if quad is None:
                continue
            projected = _project_eight(quad, canonical)
            if projected is not None:
                proposals.append((name, projected, math.inf))

        scored = []
        for name, candidate, candidate_residual in proposals:
            if not _plausible_quad(candidate[:4], width, height):
                continue
            if field is None:
                scored.append((0.0, name, candidate, candidate_residual))
                continue
            outer_ok, outer_sides = edge_support(field, candidate[:4])
            inner_ok, inner_sides = edge_support(field, candidate[4:])
            seen = [v for v, n in ((outer_ok, outer_sides), (inner_ok, inner_sides))
                    if n >= config.minimum_stations]
            quality = float(min(seen)) if seen else 0.0
            # Two independent measurements that agree beat one extrapolated
            # from the other, so a cross-checked proposal starts slightly ahead
            # and only loses if the image clearly prefers a rival.
            if name == "both":
                if candidate_residual <= config.review_residual:
                    quality += config.cross_check_bonus
                else:
                    # The rings contradict each other, so this proposal is a
                    # compromise between two readings at most one of which is
                    # right. Any single-ring reading is preferable; keep it only
                    # if nothing else survived, so the mismatch is still logged.
                    quality -= 1.0
            scored.append((quality, name, candidate, candidate_residual))

        if not scored:
            continue
        scored.sort(key=lambda row: -row[0])
        _, anchor, points, residual = scored[0]

        points = normalise_ring_order(points)

        if not _plausible_quad(points[:4], width, height):
            continue
        side = mean_side_length(points[:4])
        observed = points[:4] if anchor != "inner" else points[4:]
        clipped = bool(
            (observed[:, 0] < 2).any()
            or (observed[:, 1] < 2).any()
            or (observed[:, 0] > width - 3).any()
            or (observed[:, 1] > height - 3).any()
            or not _all_inside(points, width, height)
        )

        alignment, projected_alignment = math.inf, math.inf
        measured_ring = "outer" if anchor in ("outer", "both") else "inner"
        if field is not None:
            outer_px, outer_n = edge_alignment(field, points[:4])
            inner_px, inner_n = edge_alignment(field, points[4:])
            usable = {
                "outer": outer_px if outer_n >= config.minimum_stations else math.inf,
                "inner": inner_px if inner_n >= config.minimum_stations else math.inf,
            }
            if anchor == "both":
                seen = [v for v in usable.values() if math.isfinite(v)]
                alignment = max(seen) if seen else math.inf
                projected_alignment = alignment
            else:
                alignment = usable[measured_ring]
                projected_alignment = usable[
                    "inner" if measured_ring == "outer" else "outer"
                ]

        projected_support = -1.0
        if field is not None:
            outer_ok, outer_sides = edge_support(field, points[:4])
            inner_ok, inner_sides = edge_support(field, points[4:])
            scored = [v for v, n in ((outer_ok, outer_sides), (inner_ok, inner_sides))
                      if n >= config.minimum_stations]
            support = float(min(scored)) if scored else 0.0
            # For a single-anchor label the other ring was never measured, so
            # it is the only thing that can independently confirm the guess.
            # Taking the best measurable ring lets the anchor vouch for itself,
            # which is not evidence at all.
            if anchor == "outer":
                projected_support = (
                    float(inner_ok) if inner_sides >= config.minimum_stations else -1.0
                )
            elif anchor == "inner":
                projected_support = (
                    float(outer_ok) if outer_sides >= config.minimum_stations else -1.0
                )
        else:
            support = 0.0

        instance = GateInstance(
            points[:4], points[4:], residual, side, clipped,
            support=support, projected_support=projected_support, anchor=anchor,
        )
        # When the label cannot be checked at all, or sits badly, try solving
        # against every edge BOTH rings still show. A gate the frame has cut
        # may show only two sides of its outer square and two of its opening -
        # not four of either, so neither ring can be traced alone - yet those
        # four sides together still pin down one homography. Applied blindly
        # this degrades good labels, so it runs only where the normal path has
        # failed, and is kept only when it measurably improves the fit.
        if (field is not None and config.rescue_unmeasured
                and (not math.isfinite(alignment)
                     or alignment > config.maximum_alignment_px)):
            rescued = refine_pose_jointly(field, points)
            if rescued is not None:
                r_outer, r_outer_n = edge_alignment(field, rescued[:4])
                r_inner, r_inner_n = edge_alignment(field, rescued[4:])
                r_seen = [v for v, n in ((r_outer, r_outer_n), (r_inner, r_inner_n))
                          if n >= config.minimum_stations]
                rescued_alignment = max(r_seen) if r_seen else math.inf
                better = (
                    math.isfinite(rescued_alignment)
                    and (not math.isfinite(alignment)
                         or rescued_alignment < alignment - 0.2)
                )
                if better:
                    points = normalise_ring_order(rescued)
                    alignment = rescued_alignment
                    projected_alignment = rescued_alignment
                    instance_note = "rescued"
                else:
                    instance_note = ""
            else:
                instance_note = ""
        else:
            instance_note = ""

        instance.alignment = alignment
        if field is not None:
            instance.verified = verified_keypoints(
                field, points,
                _alignment_tolerance(side, config),
                config.minimum_side_stations,
            )
        if instance_note:
            instance.notes.append(instance_note)
        instance.outer = points[:4]
        instance.inner = points[4:]
        # When the other ring is off-frame, stop claiming it instead of
        # throwing the whole gate away. The measured ring is still a real,
        # checked observation and a gate filling the view is the last thing
        # that should go unlabelled.
        #
        # But only when the frame is what took the ring away. A gate wholly
        # inside the image has no excuse for an opening that cannot be
        # measured - and the thing that most often looks like one is the
        # course's own AI-GP signage board, which carries the same wordmark,
        # hexagon columns and sponsor row on a flat panel with no hole at all.
        # It defeats appearance matching precisely because it is the same
        # artwork. The hole is what separates a gate from a picture of one.
        if anchor != "both" and not math.isfinite(projected_alignment):
            if clipped:
                instance.suppress = (
                    "inner" if measured_ring == "outer" else "outer"
                )
            else:
                instance.flags.append("no_opening")
        claimed_visibility = keypoint_visibility(
            points, width, height, side, config.trust_reach, instance.suppress,
            instance.verified,
        )

        # Corners have to be spread like the gate they describe. Per-corner
        # verification asks whether the image has an edge at each point, and
        # unrelated structure - a pole, a gate standing further away - can
        # satisfy that. A label asserting a large gate while huddling its
        # claimed corners into a small patch has found edges belonging to
        # something else. Healthy labels span about 1.16 of their own gate
        # side; these failures run 0.08 to 0.11.
        #
        # The gate is still there, so drop the corners rather than the gate:
        # what remains is a box, which asserts the thing exists without
        # teaching a corner that sits on a pole.
        spread_ok = True
        if int((claimed_visibility > 0).sum()) >= 2 and side > 1:
            spread = points[claimed_visibility > 0]
            span = max(float(np.ptp(spread[:, 0])), float(np.ptp(spread[:, 1])))
            spread_ok = span / side >= config.minimum_corner_spread
        if not spread_ok:
            instance.verified = np.zeros(8, bool)
            instance.notes.append("corners_scattered")
            claimed_visibility = keypoint_visibility(
                points, width, height, side, config.trust_reach,
                instance.suppress, instance.verified,
            )
        claimed = int((claimed_visibility > 0).sum())

        if side < config.minimum_outer_side_px:
            instance.reason = "too_small"
        elif claimed == 0 and not _plausible_extent(points, width, height):
            instance.reason = "nothing_seen"
        elif anchor == "both" and residual > config.review_residual:
            # Both rings measured and they disagree about the gate's geometry,
            # and support could not tell them apart. Not a gate we located.
            instance.reason = "opening_mismatch"
        else:
            # A label only claims the points it writes down. Corners carried too
            # far past the frame are already written unknown, so being clipped
            # is no longer a reason to doubt what IS claimed - only the claimed
            # part has to earn acceptance.
            # Accuracy is judged in pixels, which is fair to a gate of any size
            # and to one the frame has cut. The old hit-rate scored a clipped
            # gate at half a clean one purely because it had fewer stations to
            # spend, which held back exactly the close gates worth having.
            # Near edge-on, the opening stops being an opening: the quad can
            # settle on the gate's side face or its hexagon pattern and still
            # sit on perfectly real edges, so alignment cannot catch it. The
            # learned-appearance check cannot either, because too little of the
            # face is on screen for it to run.
            sides = [float(np.linalg.norm(points[i] - points[(i + 1) % 4]))
                     for i in range(4)]
            aspect = max(sides) / max(min(sides), 1e-6)
            if aspect > config.maximum_aspect:
                instance.flags.append("edge_on")
            elif aspect > config.maximum_aspect * 0.7:
                instance.notes.append(f"oblique{aspect:.1f}")

            # Accuracy is measured, not inferred. A label is judged by how far
            # its sides sit from the image's own edges - in absolute pixels,
            # and also relative to the gate, so a small gate still has to be
            # proportionally right rather than merely close in pixel terms.
            allowed = _alignment_tolerance(side, config)
            confirmed = int((claimed_visibility > 0).sum())
            instance.notes.append(f"kp{confirmed}")
            if confirmed < config.minimum_confirmed_keypoints:
                # Not enough confirmed corners for pose supervision - but the
                # gate is still there. Written as a box with no keypoints it
                # teaches the detector that this is a gate without teaching it
                # corners nobody checked. Dropping it instead would teach the
                # opposite: that a gate filling the view is background.
                #
                # Keep whatever corners the image did verify. On a gate the
                # drone is about to fly through, two confirmed opening corners
                # are not a rounding error - they are the edge it has to steer
                # between, and they are measured, not guessed. Flag the label
                # as too thin for pose supervision, but do not delete evidence
                # that is sitting there in the picture.
                instance.flags.append("box_only")
            if not math.isfinite(alignment):
                instance.flags.append("unverified")
            elif alignment > allowed:
                instance.flags.append("off_edge")

            if anchor != "both":
                instance.notes.append(f"{anchor}_only")
                if instance.suppress:
                    instance.notes.append(f"no_{instance.suppress}")
                elif projected_alignment > config.maximum_alignment_px:
                    # The projection is measurable and it misses. Drop that
                    # ring rather than the gate - the measured ring is still a
                    # real observation, and per-corner verification below will
                    # in any case not claim anything the image does not back.
                    instance.notes.append("projection_dropped")
            elif residual > config.accept_residual:
                # The rings disagreed before rectification. Measured against
                # the image afterwards these come out at ~2.5px, so this is a
                # description of how the label was reached, not a defect - the
                # alignment check above already decides whether it is accurate.
                instance.notes.append(f"soft{residual:.3f}")
            if clipped:
                instance.notes.append("clipped")
            if claimed < 8:
                instance.notes.append(f"seen{claimed}")
            if side < config.minimum_outer_side_px * 2.0:
                # Being small is not being wrong. Every one of these measured
                # inside tolerance, so size is recorded and the relative cap
                # above does the gatekeeping.
                instance.notes.append("small")
        instance.notes.append(f"sup{support:.2f}")

        # An opening has to be an opening. A stretch of a gate's own bar passes
        # every other check - same orange, same artwork, a plausible quad - but
        # what it encloses is more gate, not the room beyond. Measured on
        # fully-confirmed gates this sits at 0.002 typical and 0.33 at the
        # 99th percentile, while the bar fragments that prompted this come in
        # at 0.34 and 0.38.
        opening_orange, _ = opening_is_see_through(image, mask, instance.inner)
        instance.notes.append(f"thru{opening_orange:.2f}")
        if opening_orange > config.maximum_opening_orange and not instance.reason:
            instance.flags.append("opening_not_open")

        # Last gate: does this even look like a gate? Edge support only asks
        # whether edges exist where the label says. A door, a banner edge or a
        # equipment panel can satisfy that and still be nothing like a gate.
        # The learned face settles it - but only when enough of the face is
        # actually on screen to judge.
        if appearance is not None and not instance.reason:
            match, coverage = appearance.score(gray, instance.outer)
            instance.appearance = match
            if coverage >= config.appearance_min_coverage:
                instance.notes.append(f"app{match:.2f}")
                if match < config.appearance_threshold:
                    instance.flags.append("unlike_gate")
        instances.append(instance)

    # A gate's own frame can look like a small gate. The bars carry the same
    # orange and the same hexagon artwork, so a stretch of one can pass every
    # per-instance check while being a piece of a gate already labelled.
    #
    # Geometry separates the two cases exactly. A genuine gate standing behind
    # another is seen THROUGH its opening, so it falls inside the inner ring.
    # A piece of bar falls between the rings - inside the outer square but
    # outside the opening - which is precisely where no real gate can be.
    live = [item for item in instances if item.verdict != "rejected"]
    for small in live:
        for large in live:
            if large is small or large.outer_side_px < 2.2 * small.outer_side_px:
                continue
            center = small.points.mean(axis=0).astype(np.float32)
            outer_poly = large.outer.astype(np.float32).reshape(-1, 1, 2)
            inner_poly = large.inner.astype(np.float32).reshape(-1, 1, 2)
            in_outer = cv2.pointPolygonTest(outer_poly, (float(center[0]), float(center[1])), False) >= 0
            in_inner = cv2.pointPolygonTest(inner_poly, (float(center[0]), float(center[1])), False) >= 0
            if in_outer and not in_inner:
                small.flags.append("on_gate_frame")
                break

    # A big orange region that produced nothing labelable is the dangerous case:
    # written as an empty label file it becomes a confident "no gate here" on a
    # frame dominated by a gate. Report it so the caller can quarantine the
    # frame instead of teaching the detector to ignore gates at close range.
    if not [item for item in instances if item.verdict != "rejected"]:
        biggest = max(
            (cv2.contourArea(contour)
             for index, contour in enumerate(contours) if hierarchy[index][3] == -1),
            default=0.0,
        )
        if biggest > config.unlabelable_area_fraction * width * height:
            instances.append(
                GateInstance(
                    np.zeros((4, 2), np.float32), np.zeros((4, 2), np.float32),
                    math.inf, math.sqrt(biggest), True,
                    anchor="none", reason="unlabelable",
                )
            )
    return instances


def _best_corners(field, gray, quad: np.ndarray, config: AutolabelConfig) -> np.ndarray:
    """Edge re-fit when the sides are long enough to measure, else corner-local."""
    if field is not None:
        current = order_corners(quad)
        for pass_index in range(max(config.edge_refine_passes, 1)):
            # Later passes search a tighter band: the first pass does the
            # travelling, the rest only polish.
            refined = refine_edges_subpixel(
                field, current,
                config.edge_refine_radius / (1.0 + pass_index),
                proximity_frac=config.edge_proximity_frac,
                size_frac=config.edge_size_frac,
            )
            if refined is None:
                break
            current = refined
        if current is not quad:
            return current
    return order_corners(refine_subpixel(gray, quad, config.subpix_window))


def _distinct_apertures(options: list[np.ndarray]) -> list[np.ndarray]:
    """Collapse the quads that describe one opening; keep one per opening."""
    kept: list[np.ndarray] = []
    for quad in options:
        center = quad.mean(axis=0)
        span = mean_side_length(quad)
        if any(
            np.linalg.norm(center - other.mean(axis=0))
            < 0.6 * max(span, mean_side_length(other))
            for other in kept
        ):
            continue
        kept.append(quad)
    return kept


def _alignment_tolerance(side: float, config: "AutolabelConfig") -> float:
    """How far a side may sit from the image's edge, for a gate this size.

    A flat pixel budget quietly punishes the close gates. Four pixels is a
    generous 3% of a 130px gate and a punishing 0.4% of a 1000px one, whose
    edges are wider and softer simply because it is nearer. So the floor grows
    slowly with the gate while the relative cap still holds the small ones to
    account - and the close-range gates, the ones being flown through, stop
    failing a test calibrated for distant ones.
    """
    floor = max(config.maximum_alignment_px, config.alignment_size_share * side)
    return min(floor, config.relative_alignment * side) if side > 0 else config.maximum_alignment_px


def _plausible_extent(points: np.ndarray, width: int, height: int) -> bool:
    """Is any worthwhile part of this gate actually on screen?"""
    xs = np.clip(points[:, 0], 0, width - 1)
    ys = np.clip(points[:, 1], 0, height - 1)
    return bool((xs.max() - xs.min()) > 24 and (ys.max() - ys.min()) > 24)


def _touches_border(contour: np.ndarray, width: int, height: int, margin: int = 3) -> bool:
    points = contour.reshape(-1, 2)
    return bool(
        (points[:, 0] <= margin).any() or (points[:, 1] <= margin).any()
        or (points[:, 0] >= width - 1 - margin).any()
        or (points[:, 1] >= height - 1 - margin).any()
    )


def _plausible_quad(quad: np.ndarray, width: int, height: int) -> bool:
    """Reject the blow-ups that near-parallel sides produce when grouping slips."""
    diagonal = math.hypot(width, height)
    center = np.array([width / 2.0, height / 2.0])
    if (np.linalg.norm(quad - center, axis=1) > 3.5 * diagonal).any():
        return False
    sides = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
    if min(sides) < 1e-3 or max(sides) / min(sides) > 6.0:
        return False
    return True


def _all_inside(points: np.ndarray, width: int, height: int) -> bool:
    return bool(
        (points[:, 0] >= 0).all() and (points[:, 1] >= 0).all()
        and (points[:, 0] < width).all() and (points[:, 1] < height).all()
    )


def keypoint_visibility(
    points: np.ndarray, width: int, height: int, side: float,
    trust_reach: float, suppress: str = "", verified: np.ndarray | None = None,
) -> np.ndarray:
    """Grade each keypoint: measured, extrapolated-but-close, or unknowable.

    Measured against synthetic crops of gates this detector reads confidently,
    a corner carried off the frame is wrong by ~6% of gate side within 0.15 of
    a side, ~17% by 0.3, and ~44% by 0.5. So a little extrapolation is worth
    keeping and a lot is worth admitting to. Beyond the trust radius the point
    is written unknown rather than confidently wrong - a bad corner reaches PnP
    with full weight, while a missing one simply is not used.
    """
    inside = (
        (points[:, 0] >= 0) & (points[:, 0] < width)
        & (points[:, 1] >= 0) & (points[:, 1] < height)
    )
    dx = np.maximum(np.maximum(-points[:, 0], points[:, 0] - (width - 1)), 0.0)
    dy = np.maximum(np.maximum(-points[:, 1], points[:, 1] - (height - 1)), 0.0)
    reach = np.hypot(dx, dy) / max(side, 1e-6)
    visibility = np.where(inside, 2, np.where(reach <= trust_reach, 1, 0))
    visibility = visibility.astype(int)
    # A ring the image cannot confirm is not written down. Better an honest gap
    # than eight confident numbers of which four were never checked.
    # NOTE: `suppress` predates per-corner verification and is no longer
    # applied. It discarded a whole ring because the ring as a whole could not
    # be checked, which threw away corners that were individually measurable -
    # on close gates that cost 131 on-screen outer corners, and with them the
    # fourth point PnP needs. Per-corner verification below already refuses
    # anything the image does not back, one corner at a time.
    if verified is not None:
        # Claim only the corners the image vouches for. A corner carried a
        # little past the frame is fine when the sides forming it were both
        # measured; one whose sides were not is a guess, and a guess written
        # down with full confidence is worse than an honest gap.
        visibility[~verified] = 0
    return visibility


def yolo_pose_line(
    instance: GateInstance, width: int, height: int, config: AutolabelConfig,
    class_id: int = 0,
) -> str:
    """class cx cy w h then eight (x, y, visibility) triplets, all normalised.

    Visibility follows the COCO convention: 2 seen, 1 labelled but not visible,
    0 not labelled. A gate is written whenever any point survives - a partly
    cropped gate is still a gate, and dropping it would teach the detector that
    a gate filling the frame is background.
    """
    points = instance.points
    visibility = keypoint_visibility(
        points, width, height, instance.outer_side_px, config.trust_reach,
        instance.suppress, instance.verified,
    )
    if not (visibility > 0).any():
        # Box-only: the gate is asserted, its corners are not. Every keypoint
        # is written unknown, so a pose head simply skips them while the
        # detector still learns the object is here.
        xs = np.clip(points[:, 0], 0.0, width - 1.0)
        ys = np.clip(points[:, 1], 0.0, height - 1.0)
        if (xs.max() - xs.min()) < 24 or (ys.max() - ys.min()) < 24:
            return ""
        fields = [
            str(class_id),
            f"{(xs.min() + xs.max()) / 2 / width:.6f}",
            f"{(ys.min() + ys.max()) / 2 / height:.6f}",
            f"{(xs.max() - xs.min()) / width:.6f}",
            f"{(ys.max() - ys.min()) / height:.6f}",
        ]
        return " ".join(fields + ["0", "0", "0"] * KEYPOINT_COUNT)

    seen = points[visibility == 2]
    if len(seen) == 0:
        seen = points[visibility == 1]
    x0, y0 = seen.min(axis=0)
    x1, y1 = seen.max(axis=0)
    x0, y0 = max(float(x0), 0.0), max(float(y0), 0.0)
    x1, y1 = min(float(x1), width - 1.0), min(float(y1), height - 1.0)
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return ""

    fields = [
        str(class_id),
        f"{(x0 + x1) / 2 / width:.6f}",
        f"{(y0 + y1) / 2 / height:.6f}",
        f"{(x1 - x0) / width:.6f}",
        f"{(y1 - y0) / height:.6f}",
    ]
    for (x, y), flag in zip(points, visibility):
        # A keypoint off the frame normalises outside [0, 1], which Ultralytics
        # rejects as a corrupt label - it silently drops the whole image, so
        # the cost of writing one is a lost frame, not a lost point. Off-frame
        # corners are therefore written unknown here. Their estimated positions
        # live in report.csv for anything (PnP, analysis) that can use a
        # coordinate outside the picture.
        if flag == 0 or not (0.0 <= x < width and 0.0 <= y < height):
            fields += ["0", "0", "0"]
        else:
            fields += [f"{x / width:.6f}", f"{y / height:.6f}", str(int(flag))]
    return " ".join(fields)


def draw_overlay(image: np.ndarray, instances: list[GateInstance]) -> np.ndarray:
    canvas = image.copy()
    colours = {
        "auto": (80, 220, 80),
        "review": (0, 180, 255),
        "rejected": (60, 60, 220),
    }
    for instance in instances:
        colour = colours[instance.verdict]
        cv2.polylines(canvas, [instance.outer.astype(np.int32)], True, colour, 3)
        cv2.polylines(canvas, [instance.inner.astype(np.int32)], True, colour, 2)
        if instance.verdict == "rejected":
            continue
        points = instance.points.astype(np.int32)
        for index, point in enumerate(points):
            cv2.circle(canvas, tuple(point), 7, (255, 0, 255), -1)
            cv2.putText(
                canvas, str(index), tuple(point + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
        label = f"{instance.residual:.3f} {'+'.join(instance.flags) or 'clean'}"
        cv2.putText(
            canvas, label, tuple(instance.outer[0].astype(int) + [0, -12]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2,
        )
    return canvas


DATA_YAML = f"""# Written by tools/autolabel_gate_pose.py.
# Dataset-local paths, class name 'gate', and a real flip_idx - the three
# things a Roboflow export usually gets wrong (models/README.md).
train: train/images
val: valid/images
kpt_shape: [{KEYPOINT_COUNT}, 3]
flip_idx: {FLIP_INDEX}
names: ['gate']
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path, help="directory of captured frames")
    parser.add_argument("--out", type=Path, default=Path("datasets/autolabel"))
    parser.add_argument("--limit", type=int, default=0, help="stop after N frames")
    parser.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    parser.add_argument(
        "--min-side", type=float, default=AutolabelConfig.minimum_outer_side_px,
        help="smallest labelable outer square, in pixels",
    )
    parser.add_argument(
        "--no-overlays", action="store_true", help="skip writing review images"
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.15,
        help="share of frames held out as the validation split (0 disables)",
    )
    parser.add_argument(
        "--copy-images", action="store_true",
        help="copy frames instead of symlinking them, so the folder zips and "
             "uploads intact",
    )
    parser.add_argument(
        "--accepted-only", action="store_true",
        help="write only auto-accepted instances; flagged ones are left out "
             "of the labels entirely rather than shipped unreviewed",
    )
    args = parser.parse_args()

    config = AutolabelConfig(minimum_outer_side_px=args.min_side)
    frames_dir = args.frames.expanduser()
    frames = sorted(
        path for path in frames_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )[:: max(args.stride, 1)]
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        parser.error(f"no images under {frames_dir}")

    out = args.out.expanduser()
    for split in ("train", "valid"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    # A deterministic, evenly spread validation split. Taking every Nth frame
    # rather than a random sample keeps both captures, every gate on the course
    # and the full range of distances represented on both sides - and taking a
    # contiguous tail would put a whole stretch of the walk in one split only.
    # Neighbouring frames are half a second apart and genuinely different
    # viewpoints, so this does not leak the way splitting 60fps video would.
    stride_for_val = max(int(round(1.0 / max(args.val_fraction, 1e-6))), 2)
    validation = {
        path.name for index, path in enumerate(frames)
        if args.val_fraction > 0 and index % stride_for_val == stride_for_val - 1
    }
    overlays = out / "overlays"
    if not args.no_overlays:
        overlays.mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text(DATA_YAML)

    # Pass one: learn what a gate face looks like from the labels this capture
    # supports best, so pass two can tell a gate from anything else that merely
    # has edges in the right places.
    appearance = None
    if config.appearance_gate:
        teachers = frames[:: max(len(frames) // 240, 1)]
        faces: list[np.ndarray] = []
        for path in teachers:
            image = cv2.imread(str(path))
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            for item in gate_candidates(image, config):
                if (item.verdict == "auto" and item.anchor == "both"
                        and not item.clipped
                        and item.support >= config.appearance_teacher_support
                        and item.outer_side_px >= config.appearance_teacher_side):
                    face = rectified_face(gray, item.outer)
                    if face is not None:
                        faces.append(face)
        appearance = learn_appearance(faces)
        print(f"  learned gate appearance from {len(faces)} strong labels"
              if appearance is not None else
              "  too few strong labels to learn an appearance; gate disabled",
              flush=True)

    rows: list[dict[str, object]] = []
    quarantined: list[str] = []
    counts = {"auto": 0, "review": 0, "rejected": 0}
    empty_frames = 0

    for position, path in enumerate(frames, 1):
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        instances = gate_candidates(image, config, appearance)
        if args.accepted_only:
            kept = [item for item in instances if item.verdict == "auto"]
        else:
            kept = [item for item in instances if item.verdict != "rejected"]

        if any(item.reason == "unlabelable" for item in instances):
            quarantined.append(path.name)
            rows.append({
                "file": path.name, "verdict": "quarantined", "residual": "inf",
                "outer_side_px": 0.0, "clipped": 1, "support": 0.0,
                "proj_support": -1.0, "appearance": -2.0, "align_px": "inf",
                "anchor": "none",
                "flags": "", "reason": "unlabelable",
            })
            continue

        lines = [
            line for line in
            (yolo_pose_line(item, width, height, config) for item in kept) if line
        ]
        split = "valid" if path.name in validation else "train"
        # An empty label file is a negative, and a detector needs negatives -
        # but only for frames we are confident are actually empty.
        (out / split / "labels" / f"{path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        target = out / split / "images" / path.name
        if not target.exists():
            if args.copy_images:
                shutil.copy2(path, target)
            else:
                target.symlink_to(path.resolve())

        if not kept:
            empty_frames += 1
        for item in instances:
            counts[item.verdict] += 1
            rows.append({
                "file": path.name,
                "verdict": item.verdict,
                "residual": round(item.residual, 4)
                if math.isfinite(item.residual) else "inf",
                "outer_side_px": round(item.outer_side_px, 1),
                "clipped": int(item.clipped),
                "support": round(item.support, 3),
                "proj_support": round(item.projected_support, 3),
                "appearance": round(item.appearance, 3),
                "align_px": round(item.alignment, 2) if math.isfinite(item.alignment) else "inf",
                "anchor": item.anchor,
                "flags": item.tags,
                "reason": item.reason,
            })

        if not args.no_overlays and instances:
            cv2.imwrite(str(overlays / path.name), draw_overlay(image, instances))
        if position % 100 == 0:
            print(f"  {position}/{len(frames)} frames", flush=True)

    with (out / "report.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "verdict", "residual", "outer_side_px",
                        "clipped", "support", "proj_support", "appearance",
                        "align_px", "anchor", "flags", "reason"],
        )
        writer.writeheader()
        # Worst first: the top of this file is the human's work queue.
        writer.writerows(sorted(
            rows,
            key=lambda row: -(row["residual"] if isinstance(row["residual"], float)
                              else 1e9),
        ))

    if quarantined:
        (out / "quarantine.txt").write_text("\n".join(quarantined) + "\n")
    labelled = counts["auto"] + counts["review"]
    print(
        f"\n{len(frames)} frames -> {labelled} gate instances "
        f"({counts['auto']} auto, {counts['review']} need review), "
        f"{counts['rejected']} rejected, {empty_frames} frames with no gate, "
        f"{len(quarantined)} quarantined (gate present, not labelable)."
    )
    train_count = len(list((out / "train" / "labels").glob("*.txt")))
    valid_count = len(list((out / "valid" / "labels").glob("*.txt")))
    print(f"Split: {train_count} train / {valid_count} valid frames.")
    print(f"Labels and data.yaml under {out}; work queue in {out / 'report.csv'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

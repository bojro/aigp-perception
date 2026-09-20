"""The gate detector as the flight code should call it on the aircraft.

Same surface as ``inference.GateDetector`` in the flight repo -- ``detect()``
returns ``Gate`` objects carrying ``aim_point`` and ``aim_offset`` -- so
swapping it in changes an import and nothing else. Two things differ, and both
are there because of something measured rather than preferred:

* It runs on onnxruntime instead of torch. Ultralytics on a Jetson means
  building torch for the platform; onnxruntime is a wheel. The provider in use
  is reported, never assumed, because a silent fall back to CPU turns a 30 fps
  budget into about 4 fps and looks only like sluggishness in the log.

* ``aim_point`` prefers the gate centre recovered by PnP. Averaging the
  corners that happen to be visible biases the target toward whichever side is
  still in frame: measured against the PnP centre, that average sits about 1%
  of the gate's width off while all four inner corners are visible, and 11-47%
  off once they are not. The corners leave frame exactly as the drone closes on
  the gate, so the cheap version is most wrong at the moment it matters most.
  Averaging remains the fallback for when PnP has nothing to say.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort

KEYPOINTS = 8
KPT_NAMES = ["out_UL", "out_UR", "out_BR", "out_BL",
             "in_UL", "in_UR", "in_BR", "in_BL"]
OUTER = slice(0, 4)
INNER = slice(4, 8)


@dataclass
class Gate:
    """One detected gate, in pixels of the frame that was handed in."""

    conf: float
    box: tuple[float, float, float, float]      # x1, y1, x2, y2
    keypoints: np.ndarray                       # (8, 2) float32
    kpt_conf: np.ndarray                        # (8,)   float32
    kpt_conf_thres: float = 0.5
    # Filled in by the detector when a pose was recovered.
    pose: Optional[object] = None
    kpt_visible: np.ndarray = field(init=False)

    def __post_init__(self):
        self.kpt_visible = (
            (self.kpt_conf >= self.kpt_conf_thres)
            & (self.keypoints.sum(axis=1) > 0)
        )

    @property
    def outer(self) -> np.ndarray:
        return self.keypoints[OUTER]

    @property
    def inner(self) -> np.ndarray:
        return self.keypoints[INNER]

    @property
    def box_centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def aim_point(self) -> tuple[float, float]:
        """Where to fly: the centre of the opening, as well as it is known.

        A solved pose gives the true centre even when most of the gate has
        left the frame. Without one, fall back to averaging whichever corners
        are visible, which is right while the ring is whole and increasingly
        wrong as it is cut.
        """
        centre = getattr(self.pose, "centre_px", None)
        if centre is not None:
            return float(centre[0]), float(centre[1])
        for pts, vis in ((self.inner, self.kpt_visible[INNER]),
                         (self.outer, self.kpt_visible[OUTER])):
            if vis.sum() >= 2:
                cx, cy = pts[vis].mean(axis=0)
                return float(cx), float(cy)
        return self.box_centre

    @property
    def aim_is_from_pose(self) -> bool:
        return getattr(self.pose, "centre_px", None) is not None

    def aim_offset(self, frame_shape: Sequence[int]) -> tuple[float, float]:
        """Aim point about the image centre, in [-1, 1] (x right, y down)."""
        h, w = frame_shape[:2]
        ax, ay = self.aim_point
        return (ax - w / 2) / (w / 2), (ay - h / 2) / (h / 2)

    @property
    def range_m(self) -> float:
        return float(getattr(self.pose, "range_m", float("inf")))


def make_session(path: Path, prefer: str = "auto") -> ort.InferenceSession:
    """Open the model on the best provider available, preferring TensorRT."""
    available = ort.get_available_providers()
    order: list = []
    if prefer in ("auto", "trt") and "TensorrtExecutionProvider" in available:
        order.append(("TensorrtExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(Path(path).parent / "trt_cache"),
        }))
    if prefer in ("auto", "cuda") and "CUDAExecutionProvider" in available:
        order.append("CUDAExecutionProvider")
    order.append("CPUExecutionProvider")
    return ort.InferenceSession(str(path), providers=order)


def letterbox(image: np.ndarray, size: int = 640):
    """Pad to square without stretching - the pose model is trained that way."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
    return canvas, scale, left, top


def decode(output, scale, pad_x, pad_y, conf_threshold, iou_threshold=0.45):
    """[1, 4+1+K*3, N] -> boxes (xyxy) and keypoints in original-frame pixels."""
    pred = output[0].T
    pred = pred[pred[:, 4] > conf_threshold]
    if not len(pred):
        return [], [], []
    scores = pred[:, 4]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    xywh = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(),
                           conf_threshold, iou_threshold)
    if idx is None or len(idx) == 0:
        return [], [], []
    boxes, kpts, confs = [], [], []
    for i in np.array(idx).ravel():
        kp = pred[i, 5:5 + KEYPOINTS * 3].reshape(KEYPOINTS, 3).copy()
        kp[:, 0] = (kp[:, 0] - pad_x) / scale
        kp[:, 1] = (kp[:, 1] - pad_y) / scale
        x, y, w, h = xywh[i]
        x1, y1 = (x - pad_x) / scale, (y - pad_y) / scale
        boxes.append((float(x1), float(y1),
                      float(x1 + w / scale), float(y1 + h / scale)))
        kpts.append(kp)
        confs.append(float(scores[i]))
    return boxes, kpts, confs


class GateDetector:
    """Pose model plus, optionally, the PnP solve that makes the aim honest.

    ``pose_solver`` is injected rather than imported so this module stays
    usable without the flight repo on the path. It is handed the (8, 2)
    keypoints and (8,) confidences in frame pixels along with the frame size,
    and returns anything carrying ``centre_px`` and ``range_m``, or None.
    """

    def __init__(self, weights: str | Path,
                 imgsz: int = 640, conf: float = 0.4, iou: float = 0.45,
                 kpt_conf: float = 0.5, provider: str = "auto",
                 pose_solver: Optional[Callable] = None, warmup: bool = True):
        self.session = make_session(Path(weights), provider)
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.kpt_conf = kpt_conf
        self.pose_solver = pose_solver
        if warmup:
            self.detect(np.zeros((imgsz, imgsz, 3), np.uint8))

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    @property
    def on_cpu(self) -> bool:
        return self.provider == "CPUExecutionProvider"

    def detect(self, frame: np.ndarray) -> list[Gate]:
        """Run one BGR frame. Gates come back best-first (see ``rank``)."""
        canvas, scale, pad_x, pad_y = letterbox(frame, self.imgsz)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: blob})[0]
        boxes, kpts, confs = decode(out, scale, pad_x, pad_y, self.conf, self.iou)

        gates = []
        for box, kp, c in zip(boxes, kpts, confs):
            g = Gate(c, box, kp[:, :2].astype(np.float32),
                     kp[:, 2].astype(np.float32), self.kpt_conf)
            if self.pose_solver is not None:
                try:
                    g.pose = self.pose_solver(g.keypoints, g.kpt_conf,
                                              frame.shape, c, box)
                except Exception:
                    # A failed solve must degrade to the averaged aim point,
                    # never take the vision thread down mid-flight.
                    g.pose = None
            gates.append(g)
        gates.sort(key=self.rank, reverse=True)
        return gates

    @staticmethod
    def rank(g: Gate) -> float:
        """Nearest, most confident gate first: that is the one being flown at."""
        return g.conf * float(np.sqrt(g.area))

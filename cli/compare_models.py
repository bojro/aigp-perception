"""Run two gate-pose models on the same frames, side by side.

Built for the Orin, so it needs only onnxruntime and OpenCV - no torch, which
is the awkward install on a Jetson. It reports what each model detects, how far
apart their keypoints are and how long each takes, and writes a side-by-side
image per frame so the difference can be judged by eye.

The provider actually in use is printed, because a silent fall back to CPU
makes the timings meaningless.

    python cli/compare_models.py --a models/gate_pose_teammate.onnx \
        --b models/gate_pose_hybrid_v1.onnx --frames "~/gate frames" --limit 60
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

KEYPOINTS = 8


def make_session(path: Path, prefer: str = "auto") -> ort.InferenceSession:
    available = ort.get_available_providers()
    order: list = []
    if prefer in ("auto", "trt") and "TensorrtExecutionProvider" in available:
        order.append(("TensorrtExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(path.parent / "trt_cache"),
        }))
    if prefer in ("auto", "cuda") and "CUDAExecutionProvider" in available:
        order.append("CUDAExecutionProvider")
    order.append("CPUExecutionProvider")
    return ort.InferenceSession(str(path), providers=order)


def letterbox(image: np.ndarray, size: int = 640):
    """Pad to square without stretching - the pose model is trained letterboxed."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
    return canvas, scale, left, top


def decode(output, scale, pad_x, pad_y, conf_threshold, iou_threshold=0.45):
    """[1, 4+1+K*3, N] -> boxes and keypoints back in original image pixels."""
    pred = output[0].T
    keep = pred[:, 4] > conf_threshold
    pred = pred[keep]
    if not len(pred):
        return [], []
    scores = pred[:, 4]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1)
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    if idx is None or len(idx) == 0:
        return [], []
    out_boxes, out_kpts = [], []
    for i in np.array(idx).ravel():
        kp = pred[i, 5:5 + KEYPOINTS * 3].reshape(KEYPOINTS, 3).copy()
        kp[:, 0] = (kp[:, 0] - pad_x) / scale
        kp[:, 1] = (kp[:, 1] - pad_y) / scale
        x, y, w, h = boxes[i]
        out_boxes.append(np.array([(x - pad_x) / scale, (y - pad_y) / scale,
                                   w / scale, h / scale]))
        out_kpts.append(kp)
    return out_boxes, out_kpts


def draw(image, boxes, kpts, colour, label):
    for box, kp in zip(boxes, kpts):
        x, y, w, h = box.astype(int)
        cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)
        for ring, c in ((slice(0, 4), colour), (slice(4, 8), (255, 200, 60))):
            if (kp[ring][:, 2] > 0.5).all():
                cv2.polylines(image, [kp[ring][:, :2].astype(np.int32)], True, c, 2)
        for px, py, pc in kp:
            if pc >= 0.5:
                cv2.circle(image, (int(px), int(py)), 5, (255, 0, 255), -1)
    cv2.putText(image, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 3)
    return image


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--frames", type=Path)
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=Path("compare_out"))
    ap.add_argument("--provider", default="auto", choices=("auto", "trt", "cuda", "cpu"))
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    sessions = {}
    for tag, path in (("A", args.a), ("B", args.b)):
        s = make_session(path.expanduser(), args.provider)
        sessions[tag] = s
        print(f"{tag}: {path.name}  providers={s.get_providers()}")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        files = None
    else:
        allf = sorted(p for p in args.frames.expanduser().iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        files = allf[::max(len(allf) // args.limit, 1)][:args.limit]
        cap = None

    # warm up so the first-call graph build does not land in the timings
    dummy = np.zeros((1, 3, 640, 640), np.float32)
    for s in sessions.values():
        for _ in range(args.warmup):
            s.run(None, {s.get_inputs()[0].name: dummy})

    stats = {t: {"ms": [], "dets": 0} for t in ("A", "B")}
    agree = only_a = only_b = done = 0
    gaps = []

    while done < args.limit:
        if cap is not None:
            ok, image = cap.read()
            if not ok:
                break
            name = f"cam_{done:04d}"
        else:
            if done >= len(files):
                break
            image = cv2.imread(str(files[done]))
            name = files[done].stem
            if image is None:
                done += 1
                continue

        blob, scale, px, py = letterbox(image)
        tensor = np.ascontiguousarray(
            blob[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)

        result = {}
        for tag, session in sessions.items():
            t0 = time.perf_counter()
            out = session.run(None, {session.get_inputs()[0].name: tensor})[0]
            stats[tag]["ms"].append((time.perf_counter() - t0) * 1000)
            boxes, kpts = decode(out, scale, px, py, args.conf)
            stats[tag]["dets"] += len(boxes)
            result[tag] = (boxes, kpts)

        used = set()
        for ka in result["A"][1]:
            best, bd = None, 1e9
            for j, kb in enumerate(result["B"][1]):
                if j in used:
                    continue
                d = float(np.linalg.norm(kb[:, :2].mean(0) - ka[:, :2].mean(0)))
                if d < bd:
                    best, bd = j, d
            span = max(float(np.ptp(ka[:4, 0])), float(np.ptp(ka[:4, 1])), 1.0)
            if best is not None and bd < 0.45 * span:
                used.add(best)
                agree += 1
                gaps.append(float(np.median(np.linalg.norm(
                    result["B"][1][best][:, :2] - ka[:, :2], axis=1))))
            else:
                only_a += 1
        only_b += len(result["B"][1]) - len(used)

        left = draw(image.copy(), *result["A"], (90, 230, 90), f"A  {args.a.stem}")
        right = draw(image.copy(), *result["B"], (0, 190, 255), f"B  {args.b.stem}")
        cv2.imwrite(str(args.out / f"{name}.jpg"),
                    cv2.resize(np.hstack([left, right]), (1920, 540)))
        done += 1

    print(f"\nframes: {done}")
    for tag in ("A", "B"):
        ms = np.array(stats[tag]["ms"])
        print(f"  {tag}: {stats[tag]['dets']:4d} detections | "
              f"{np.median(ms):6.1f} ms median, {np.percentile(ms, 90):6.1f} ms p90"
              f"  ({1000 / np.median(ms):5.1f} fps)")
    print(f"\n  both found   : {agree}")
    print(f"  only A found : {only_a}")
    print(f"  only B found : {only_b}")
    if gaps:
        g = np.array(gaps)
        print(f"  keypoint gap where both found: p50={np.median(g):.1f}px "
              f"p90={np.percentile(g, 90):.1f}px")
    print(f"\nside-by-side images in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

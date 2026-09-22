"""Does geometric refinement of the network's corners recover the precision?

The network finds every gate but places corners to ~4% of gate size. The
geometric fit places corners to under a pixel but only where it can trace
them. If the network's output is accurate enough to seed the refinement, the
combination should have the network's recall and the geometry's precision -
which is the whole vision system in one line.
"""
import glob, sys, time, warnings
warnings.filterwarnings("ignore")
import cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aigp_perception import paths as P
import aigp_perception.autolabel_gate_pose as A
from ultralytics import YOLO

model = YOLO(str(P.TEAMMATE_PT))
cfg = A.AutolabelConfig()
frames = P.frames()[::6]

raw, refined, refine_ms, improved, worsened = [], [], [], 0, 0
for path in frames:
    image = cv2.imread(path)
    if image is None:
        continue
    reference = [
        it for it in A.gate_candidates(image, cfg)
        if it.verdict == "auto" and it.verified is not None and it.verified.all()
    ]
    if not reference:
        continue
    field = A.orange_field(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = model.predict(image, imgsz=640, conf=0.25, verbose=False, device="mps")[0]
    if result.keypoints is None or not len(result.keypoints.xy):
        continue
    preds = [k.cpu().numpy() for k in result.keypoints.xy]

    for item in reference:
        truth = item.points
        best, best_d = None, 1e9
        for pred in preds:
            d = float(np.linalg.norm(pred.mean(axis=0) - truth.mean(axis=0)))
            if d < best_d:
                best, best_d = pred, d
        if best is None or best_d > 0.45 * item.outer_side_px:
            continue

        before = float(np.median(np.linalg.norm(best - truth, axis=1)))
        start = time.perf_counter()
        # Refine each ring the same way the labeller does, seeded by the net.
        fixed = best.copy().astype(np.float32)
        for lo in (0, 4):
            ring = A.order_corners(fixed[lo:lo + 4])
            out = A.refine_edges_subpixel(
                field, ring, radius=cfg.edge_refine_radius,
                proximity_frac=cfg.edge_proximity_frac,
                size_frac=cfg.edge_size_frac,
            )
            if out is None:
                out = A.order_corners(A.refine_subpixel(gray, ring, cfg.subpix_window))
            fixed[lo:lo + 4] = out
        refine_ms.append((time.perf_counter() - start) * 1000.0)
        after = float(np.median(np.linalg.norm(fixed - truth, axis=1)))
        raw.append(before); refined.append(after)
        improved += after < before
        worsened += after > before

raw, refined = np.array(raw), np.array(refined)
print(f"matched gates: {len(raw)}")
print(f"  network alone      : median {np.median(raw):6.2f}px   p90 {np.percentile(raw,90):6.2f}px")
print(f"  after refinement   : median {np.median(refined):6.2f}px   p90 {np.percentile(refined,90):6.2f}px")
print(f"  improved {improved}/{len(raw)}  worsened {worsened}/{len(raw)}")
print(f"  refinement cost    : {np.median(refine_ms):.2f} ms per gate (python, CPU)")

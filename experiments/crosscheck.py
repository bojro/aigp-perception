"""Measure the trained model against the geometric labels, on unseen frames.

The model's own validation split came from a walk-around shot every half
second, so neighbouring frames are near-duplicates and a random split leaks
training frames into validation. This capture is a different session the model
never saw, and the fully-confirmed geometric labels - every corner backed by
two independently measured sides - are a reference it had no part in making.
"""
from pathlib import Path
import csv, glob, sys, warnings
warnings.filterwarnings("ignore")
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aigp_perception.autolabel_gate_pose as A
from ultralytics import YOLO

SP = str(Path(__file__).resolve().parents[1] / "work")
model = YOLO("/Users/bojro/Downloads/best.pt")
cfg = A.AutolabelConfig()

frames = sorted(glob.glob("/Users/bojro/Downloads/gate frames/*.jpg"))[::6]
matched, missed_by_model, extra_by_model, ref_total = [], 0, 0, 0

for path in frames:
    image = cv2.imread(path)
    if image is None:
        continue
    reference = [
        it for it in A.gate_candidates(image, cfg)
        if it.verdict == "auto" and it.verified is not None and it.verified.all()
    ]
    result = model.predict(image, imgsz=640, conf=0.25, verbose=False, device="mps")[0]
    predictions = []
    if result.keypoints is not None and len(result.keypoints.xy):
        for k in range(len(result.keypoints.xy)):
            predictions.append(result.keypoints.xy[k].cpu().numpy())

    ref_total += len(reference)
    used = set()
    for item in reference:
        truth = item.points
        best, best_distance = None, 1e9
        for index, pred in enumerate(predictions):
            if index in used:
                continue
            distance = float(np.linalg.norm(pred.mean(axis=0) - truth.mean(axis=0)))
            if distance < best_distance:
                best, best_distance = index, distance
        if best is None or best_distance > 0.45 * item.outer_side_px:
            missed_by_model += 1
            continue
        used.add(best)
        errors = np.linalg.norm(predictions[best] - truth, axis=1)
        matched.append((errors, item.outer_side_px, item.clipped))
    extra_by_model += len(predictions) - len(used)

print(f"frames {len(frames)} | geometric reference gates {ref_total}")
print(f"  model matched   : {len(matched)}")
print(f"  model MISSED    : {missed_by_model}  ({100*missed_by_model/max(ref_total,1):.0f}% of verified gates)")
print(f"  model extra dets: {extra_by_model}")
if matched:
    per = np.concatenate([e for e, _, _ in matched])
    sides = np.array([s for _, s, _ in matched])
    rel = np.concatenate([e / s for e, s, _ in matched])
    print(f"\nkeypoint error vs geometric reference (n={len(per)} keypoints):")
    print(f"  absolute px : p50={np.median(per):6.1f}  p90={np.percentile(per,90):6.1f}")
    print(f"  as %% of gate: p50={100*np.median(rel):5.2f}%  p90={100*np.percentile(rel,90):5.2f}%")
    outer = np.concatenate([e[:4] for e, _, _ in matched])
    inner = np.concatenate([e[4:] for e, _, _ in matched])
    print(f"  outer ring px p50={np.median(outer):.1f} | inner ring px p50={np.median(inner):.1f}")
    for lo, hi in ((0, 250), (250, 500), (500, 1200), (1200, 9999)):
        m = (sides >= lo) & (sides < hi)
        if m.sum() < 3:
            continue
        e = np.concatenate([matched[i][0] for i in np.where(m)[0]])
        print(f"  gates {lo:5d}-{hi:<5d} n={int(m.sum()):3d}  err p50={np.median(e):6.1f}px")

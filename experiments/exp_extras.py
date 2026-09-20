"""What are the model's extra detections, and can geometry adjudicate them?

The model finds far more gates than the geometric engine can verify. Either
they are real gates the geometry could not trace - in which case the model can
seed it and recover labels - or they are false positives, in which case the
geometry can filter them. The same test settles both, and it is the test the
runtime would use anyway.
"""
import glob, sys, warnings
warnings.filterwarnings("ignore")
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aigp_perception.autolabel_gate_pose as A
from ultralytics import YOLO

model = YOLO("/Users/bojro/Downloads/best.pt")
cfg = A.AutolabelConfig()
frames = sorted(glob.glob("/Users/bojro/Downloads/gate frames/*.jpg"))[::6]

verified_hit = recovered = filtered_out = no_refine = 0
rec_align, rec_conf, rej_conf = [], [], []

for path in frames:
    image = cv2.imread(path)
    if image is None: continue
    field = A.orange_field(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = A.orange_mask(image, cfg)
    reference = [it for it in A.gate_candidates(image, cfg)
                 if it.verdict == "auto" and it.verified is not None and it.verified.all()]
    result = model.predict(image, imgsz=640, conf=0.25, verbose=False, device="mps")[0]
    if result.keypoints is None or not len(result.keypoints.xy): continue
    preds = [k.cpu().numpy() for k in result.keypoints.xy]
    confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(preds))

    for pred, conf in zip(preds, confs):
        side = A.mean_side_length(A.order_corners(pred[:4].astype(np.float32)))
        # already covered by a verified geometric label?
        if any(np.linalg.norm(pred.mean(0) - it.points.mean(0)) < 0.45 * it.outer_side_px
               for it in reference):
            verified_hit += 1
            continue
        if side < 60:
            continue
        # refine the network's corners, then judge them the way the labeller does
        fixed = pred.copy().astype(np.float32)
        ok = True
        for lo in (0, 4):
            ring = A.order_corners(fixed[lo:lo+4])
            out = A.refine_edges_subpixel(field, ring, cfg.edge_refine_radius,
                                          proximity_frac=cfg.edge_proximity_frac,
                                          size_frac=cfg.edge_size_frac)
            if out is None:
                ok = False; break
            fixed[lo:lo+4] = out
        if not ok:
            no_refine += 1; rej_conf.append(conf); continue
        allowed = min(cfg.maximum_alignment_px, cfg.relative_alignment * side)
        ver = A.verified_keypoints(field, fixed, allowed, cfg.minimum_side_stations)
        thru, _ = A.opening_is_see_through(image, mask, fixed[4:])
        if int(ver.sum()) >= 4 and thru <= cfg.maximum_opening_orange:
            recovered += 1
            rec_align.append(A.edge_alignment(field, fixed[4:])[0]); rec_conf.append(conf)
        else:
            filtered_out += 1; rej_conf.append(conf)

print(f"frames {len(frames)}")
print(f"  detections matching a verified geometric label : {verified_hit}")
print(f"  extra detections RECOVERED as new labels       : {recovered}")
print(f"  extra detections FILTERED OUT by geometry      : {filtered_out}")
print(f"  extra detections geometry could not refine     : {no_refine}")
if rec_align:
    a = np.array([x for x in rec_align if np.isfinite(x)])
    print(f"\n  recovered labels align to {np.median(a):.2f}px (median), p90 {np.percentile(a,90):.2f}px")
if rec_conf and rej_conf:
    print(f"  model confidence: recovered p50={np.median(rec_conf):.2f} | "
          f"rejected p50={np.median(rej_conf):.2f}")

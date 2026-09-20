"""Score gate-pose models the way the flight code consumes them.

Not mAP. mAP is measured against labels, and the labels here are ~4.36 px from
the humans who drew them, so a model can move within the noise and the number
will still move. Worse, our auto-labels were only ever self-scored -- the
residual was computed against the gate's own geometric model, which is the very
thing the labeller solves for, so a systematic error fits perfectly and reads
as zero. Three things are reported instead:

  PnP solve rate and reprojection error   Label-free. It is what vision/yolo_pnp.py
                                          actually feeds the policy, so a model
                                          that detects well but solves badly is
                                          caught here and nowhere else.

  Keypoint error against HUMAN labels     Real ground truth, on the held-out
                                          blocks only. Carries a human-jitter
                                          floor: differences smaller than a
                                          couple of pixels mean nothing.

  Everything bucketed by gate size        A gate filling >30% of the frame is the
                                          one the drone is about to fly through.
                                          The current hybrid is near-blind to 28%
                                          of those -- median score 0.078 on the
                                          ones it misses, which is a hole rather
                                          than a threshold. That bucket is the
                                          headline.
"""
from __future__ import annotations

import argparse, json, sys, collections
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_labels(lab_dir: Path) -> dict[str, np.ndarray]:
    """stem -> (N, 8, 3) keypoints in normalised coords, plus bbox area."""
    out = {}
    for p in sorted(lab_dir.glob("*.txt")):
        rows = []
        for line in p.read_text().split("\n"):
            f = line.split()
            if len(f) != 29:
                continue
            rows.append([float(x) for x in f[1:]])
        if rows:
            out[p.stem] = np.array(rows, dtype=np.float64)
    return out


def match(pred_c: np.ndarray, truth_c: np.ndarray, span: float) -> int | None:
    """Nearest prediction centre within 45% of gate span, as the repo does."""
    if not len(pred_c):
        return None
    d = np.linalg.norm(pred_c - truth_c, axis=1)
    j = int(np.argmin(d))
    return j if d[j] < 0.45 * span else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True,
                    help="label dir holding HUMAN labels for these frames")
    ap.add_argument("--hand-stems", type=Path, required=True,
                    help="json list of stems a human actually annotated")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=Path("eval_out"))
    args = ap.parse_args()

    from ultralytics import YOLO
    from deploy.gate_pose import solve_gate_pose

    hand = set(json.loads(args.hand_stems.read_text()))
    truth = {k: v for k, v in load_labels(args.truth).items() if k in hand}
    frames = sorted(p for p in args.images.glob("*.jpg"))
    print(f"{len(frames)} frames, {len(truth)} of them human-labelled "
          f"({sum(len(v) for v in truth.values())} gates)\n", flush=True)

    BUCKETS = [(0.0, 0.05, "tiny  <5%"), (0.05, 0.15, "mid  5-15%"),
               (0.15, 0.30, "near 15-30%"), (0.30, 1.01, "CLOSE >30%")]
    args.out.mkdir(parents=True, exist_ok=True)

    for w in args.weights:
        # runs/<tag>/weights/best.pt -> <tag>; a loose checkpoint -> its stem.
        wp = Path(w)
        name = wp.parent.parent.name if wp.parent.name == "weights" else wp.stem
        model = YOLO(w)
        dets = solves = 0
        reproj, kperr = [], []
        found = collections.Counter(); total = collections.Counter()
        kp_by_bucket = collections.defaultdict(list)

        for fp in frames:
            res = model.predict(str(fp), imgsz=640, conf=args.conf,
                                verbose=False, device=0)[0]
            H, W = res.orig_shape
            kxy = (res.keypoints.xy.cpu().numpy()
                   if res.keypoints is not None else np.zeros((0, 8, 2)))
            kcf = (res.keypoints.conf.cpu().numpy()
                   if (res.keypoints is not None and res.keypoints.conf is not None)
                   else np.ones(kxy.shape[:2]))
            dets += len(kxy)
            for k, c in zip(kxy, kcf):
                pose = solve_gate_pose(k, c, (H, W))
                if pose is not None:
                    solves += 1
                    reproj.append(pose.reproj_err_px)

            if fp.stem in truth:
                pc = kxy[:, :, :2].mean(axis=1) if len(kxy) else np.zeros((0, 2))
                for row in truth[fp.stem]:
                    kp = row[4:].reshape(8, 3)
                    vis = kp[:, 2] > 0
                    if vis.sum() < 4:
                        continue
                    tp = kp[:, :2] * [W, H]
                    span = max(np.ptp(tp[vis, 0]), np.ptp(tp[vis, 1]), 1.0)
                    area = (row[2] * row[3])
                    b = next(lab for lo, hi, lab in BUCKETS if lo <= area < hi)
                    total[b] += 1
                    j = match(pc, tp[vis].mean(axis=0), span)
                    if j is None:
                        continue
                    found[b] += 1
                    e = float(np.median(np.linalg.norm(kxy[j][vis] - tp[vis], axis=1)))
                    kperr.append(e); kp_by_bucket[b].append(e)

        r = np.array(reproj); k = np.array(kperr)
        print(f"=== {name} ===", flush=True)
        print(f"  detections {dets}, PnP solved {solves} "
              f"({solves/max(dets,1):.1%} of detections, "
              f"{solves/max(len(frames),1):.2f} per frame)", flush=True)
        if len(r):
            print(f"  reprojection  p50={np.median(r):.2f}px  p90={np.percentile(r,90):.2f}px",
                  flush=True)
        if len(k):
            print(f"  keypoint err vs HUMAN  p50={np.median(k):.2f}px  "
                  f"p90={np.percentile(k,90):.2f}px  (n={len(k)})", flush=True)
        print("  recall vs human labels, by gate size:", flush=True)
        for _, _, lab in BUCKETS:
            t = total[lab]
            if not t:
                continue
            e = kp_by_bucket[lab]
            extra = f"  kp p50={np.median(e):5.2f}px" if e else ""
            print(f"    {lab:<12} {found[lab]:4d}/{t:<4d} = {found[lab]/t:6.1%}{extra}",
                  flush=True)
        print(f"\nEVAL {name} solve_rate={solves/max(dets,1):.4f} "
              f"reproj_p50={np.median(r) if len(r) else float('nan'):.3f} "
              f"kp_p50={np.median(k) if len(k) else float('nan'):.3f} "
              f"close_recall={found['CLOSE >30%']/max(total['CLOSE >30%'],1):.4f}\n",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

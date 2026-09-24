"""The images on the README, drawn from the walk-around capture and the hybrid labels.

Each one carries one message:

  hand497_approach.gif    the shipped model, gate_pose_hand497.onnx, run on the raw frames of a walk
                          through the hall that ends close on a gate (the window is chosen by the
                          labels: the gate that grows the most); corners below the 0.25 keypoint
                          threshold left out, so what is drawn is what the flight code gets
  hand497_orbit.gif       the same model on the first hundred frames of the first session, the camera
                          circling one gate
  hand497_sway.gif        the same model holding one gate at mid range while the camera sways side to
                          side (frames 29-79 of the first session), the corners staying put
  hard_cases_labels_vs_hand497.jpg
                          four frames the pipeline is judged on (a far gate, a close gate with corners
                          off the frame, several gates at once, an oblique view), each shown twice:
                          the training label on the left, hand497's detection on the right

Needs the 1207-frame capture (AIGP_CAPTURE, default ~/Downloads/gate frames), the hybrid
dataset's YOLO-pose labels (AIGP_LABELS, default the flight repo's datasets/hybrid), and a Python with
Pillow, cv2 and onnxruntime (`pip install -e .[deploy]`).

    python3 docs/make_readme_images.py
"""
import glob, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAPTURE = os.path.expanduser(os.environ.get("AIGP_CAPTURE", "~/Downloads/gate frames"))
LABELS = os.path.expanduser(os.environ.get("AIGP_LABELS", "~/dev/ai-grand-prix/datasets/hybrid"))
OUT = os.path.dirname(os.path.abspath(__file__))
OUTER, INNER = (80, 220, 255), (255, 210, 60)      # outer ring cyan, inner ring yellow
NAMES = ["0", "1", "2", "3", "4", "5", "6", "7"]

def label_path(stem):
    for split in ("train", "valid"):
        p = os.path.join(LABELS, split, "labels", stem + ".txt")
        if os.path.exists(p): return p
    return None

def gates(stem):
    """[(box, [(x, y, visible) x 8])] in normalised coordinates."""
    p = label_path(stem)
    if not p: return []
    out = []
    for line in open(p):
        v = line.split()
        if len(v) < 5 + 24: continue
        box = tuple(map(float, v[1:5])); k = list(map(float, v[5:29]))
        out.append((box, [(k[3*i], k[3*i+1], k[3*i+2] > 0) for i in range(8)]))
    return out

def draw(im, gs, scale=1.0):
    """Draw at the image's own size: lw/r/font scale with width so overlays stay legible after resizing."""
    W, H = im.size; d = ImageDraw.Draw(im)
    k = W / 1920 * scale
    lw = max(2, round(7 * k)); r = max(3, round(11 * k))
    try: font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(9, round(26 * k)))
    except OSError: font = ImageFont.load_default()
    for box, kps in gs:
        pts = [(x * W, y * H, vis) for x, y, vis in kps]
        for base, col in ((0, OUTER), (4, INNER)):
            for a in range(4):
                i, j = base + a, base + (a + 1) % 4
                if pts[i][2] and pts[j][2]:
                    d.line([pts[i][:2], pts[j][:2]], fill=col, width=lw)
        for i, (x, y, vis) in enumerate(pts):
            if not vis: continue
            col = OUTER if i < 4 else INNER
            d.ellipse([x - r, y - r, x + r, y + r], fill=col, outline=(20, 20, 20), width=2)
            d.text((x + r + 3, y - r - 4), NAMES[i], fill="white", font=font, stroke_width=2, stroke_fill=(20, 20, 20))
    return im

def frame(stem, width):
    im = Image.open(os.path.join(CAPTURE, stem + ".jpg")).convert("RGB")
    im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    return draw(im, gates(stem))

def best_window(n=40, step=1):
    """The n-frame window that is an approach: the largest gate with six or more corners in view grows the most."""
    best, best_score = None, -1
    for prefix in ("0919_214639", "0919_220524"):
        hs = {}
        for p in glob.glob(os.path.join(LABELS, "*", "labels", prefix + "_*.txt")):
            st = os.path.splitext(os.path.basename(p))[0]; gs = gates(st)
            ok = [g for g in gs if sum(v for _, _, v in g[1]) >= 6]
            hs[int(st[-6:])] = max((g[0][3] for g in ok), default=0.0)
        for a in sorted(hs):
            win = [hs.get(a + k * step, 0.0) for k in range(n)]
            if min(win) <= 0.10: continue
            sc = win[-1] - win[0]
            if sc > best_score: best, best_score = f"{prefix}_{a:06d}", sc
    return best, best_score

def _detector(weights=None):
    import cv2  # noqa: F401
    sys.path.insert(0, os.path.join(os.path.dirname(OUT), "deploy", "onnx"))
    from gate_detector import GateDetector
    return GateDetector(weights or os.path.join(os.path.dirname(OUT), "models", "gate_pose_hand497.onnx"), conf=0.4, kpt_conf=0.25, provider="cpu")

def model_gif(name, stems, det, width=720, fps=5):
    """gate_pose_hand497.onnx on the raw frames, drawn the way the flight code would see them."""
    import cv2
    frames, n_gates, n_full = [], 0, 0
    for st in stems:
        path = os.path.join(CAPTURE, st + ".jpg")
        if not os.path.exists(path): continue
        bgr = cv2.imread(path); H, W = bgr.shape[:2]
        gs = []
        for g in det.detect(cv2.resize(bgr, (640, 360))):
            gs.append(((0, 0, 0, 0), [(x / 640, y / 360, bool(v)) for (x, y), v in zip(g.keypoints, g.kpt_visible)]))
            n_gates += 1; n_full += int(g.kpt_visible.sum() == 8)
        im = Image.fromarray(bgr[..., ::-1].copy()).resize((width, int(H * width / W)), Image.LANCZOS)
        frames.append(draw(im, gs).quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG))
    out = os.path.join(OUT, name)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(out, len(frames), "frames", round(os.path.getsize(out) / 1e6, 1), "MB;", n_gates, "gates found,", n_full, "with all 8 corners above 0.25")

def approach_gif(det, n=40):
    start, score = best_window(n=n); print("approach window", start, "gate growth", round(score, 2))
    model_gif("hand497_approach.gif", [f"{start[:-6]}{int(start[-6:]) + k:06d}" for k in range(n)], det)

def orbit_gif(det, start=0, n=50, step=2):
    model_gif("hand497_orbit.gif", [f"0919_214639_{start + k * step:06d}" for k in range(n)], det, width=600)

def sway_gif(det, start=29, n=50):
    model_gif("hand497_sway.gif", [f"0919_214639_{start + k:06d}" for k in range(n)], det, width=600, fps=6)

def pick_hard_cases(det=None):
    """Choose by the labels themselves: far, cut off, crowded, oblique. The far case also asks the model to agree
    that there is one small gate and nothing else, so an unlabelled near gate cannot dominate the frame."""
    import cv2
    stems = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(LABELS, "*", "labels", "*.txt"))]
    stats = {}
    for st in stems:
        gs = gates(st)
        if not gs or not os.path.exists(os.path.join(CAPTURE, st + ".jpg")): continue
        big = max(gs, key=lambda g: g[0][2] * g[0][3])
        vis = sum(v for _, _, v in big[1])
        stats[st] = dict(n=len(gs), h=big[0][3], vis=vis, skew=abs(big[1][0][1] - big[1][1][1]) if big[1][0][2] and big[1][1][2] else 0)
    far_cands = sorted((st for st in stats if stats[st]["vis"] == 8 and stats[st]["n"] <= 2 and 0.10 < stats[st]["h"] < 0.25), key=lambda st: stats[st]["h"])
    far = far_cands[0]
    if det is not None:
        for st in far_cands:
            gs = det.detect(cv2.resize(cv2.imread(os.path.join(CAPTURE, st + ".jpg")), (640, 360)))
            # every detection small (nothing large and unlabelled in the frame), and at least one with all eight corners
            if gs and all((g.box[3] - g.box[1]) / 360 < 0.3 for g in gs) and any(g.kpt_visible.sum() == 8 for g in gs):
                far = st; break
        else:
            print("no far-gate frame passed the model check; using", far)
    cut = max((st for st in stats if 4 <= stats[st]["vis"] <= 6), key=lambda st: stats[st]["h"])
    crowd = max((st for st in stats if stats[st]["vis"] == 8), key=lambda st: (stats[st]["n"], stats[st]["h"]))
    oblique = max((st for st in stats if stats[st]["vis"] == 8 and stats[st]["h"] > 0.3), key=lambda st: stats[st]["skew"])
    return [(far, "far gate"), (cut, "cut by the frame"), (crowd, "several gates"), (oblique, "oblique")]

def hard_cases_pairs(det, pane=700):
    """Rows: the four hard cases. Columns: the training label | hand497's detection on the raw frame."""
    import cv2
    try: font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
    except OSError: font = ImageFont.load_default()
    picks = pick_hard_cases(det); gap, strip = 6, 26
    h = int(pane * 9 / 16)
    sheet = Image.new("RGB", (2 * pane + gap, strip + 4 * (h + gap)), (13, 17, 23))
    d = ImageDraw.Draw(sheet)
    d.text((6, 5), "training label (hybrid pipeline)", fill=(220, 220, 220), font=font)
    d.text((pane + gap + 6, 5), "gate_pose_hand497.onnx detection", fill=(220, 220, 220), font=font)
    for r, (st, tag) in enumerate(picks):
        bgr = cv2.imread(os.path.join(CAPTURE, st + ".jpg")); H, W = bgr.shape[:2]
        small = Image.fromarray(bgr[..., ::-1].copy()).resize((pane, int(H * pane / W)), Image.LANCZOS)
        left = draw(small.copy(), gates(st))
        gs = [((0, 0, 0, 0), [(x / 640, y / 360, bool(v)) for (x, y), v in zip(g.keypoints, g.kpt_visible)]) for g in det.detect(cv2.resize(bgr, (640, 360)))]
        right = draw(small.copy(), gs)
        y = strip + r * (h + gap)
        sheet.paste(left, (0, y)); sheet.paste(right, (pane + gap, y))
        ImageDraw.Draw(sheet).text((8, y + 6), tag, fill="white", font=font, stroke_width=2, stroke_fill=(20, 20, 20))
        print(tag, st, len(gs), "detections")
    out = os.path.join(OUT, "hard_cases_labels_vs_hand497.jpg"); sheet.save(out, quality=86); print(out)

if __name__ == "__main__":
    det = _detector()
    approach_gif(det); orbit_gif(det); sway_gif(det); hard_cases_pairs(det)

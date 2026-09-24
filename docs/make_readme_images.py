"""The images on the README, drawn from the walk-around capture and the hybrid labels.

Each one carries one message:

  labels_walkaround.gif   the labeller returns the same eight ordered corners on every frame as the
                          camera moves around the gate; an edge is drawn only between corners that
                          were actually seen
  labels_hard_cases.jpg   four frames the pipeline is judged on: a far gate, a close gate with
                          corners off the frame, two gates at once, an oblique view

Needs the 1207-frame capture (AIGP_CAPTURE, default ~/Downloads/gate frames) and the hybrid
dataset's YOLO-pose labels (AIGP_LABELS, default the flight repo's datasets/hybrid). Pillow only.

    python3 docs/make_readme_images.py
"""
import glob, os
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
    W, H = im.size; d = ImageDraw.Draw(im)
    lw = max(2, int(5 * scale)); r = max(4, int(9 * scale))
    try: font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", int(22 * scale))
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
    im = draw(im, gates(stem))
    return im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)

def walkaround_gif(start="0919_214639_000000", n=50, step=2, width=600, fps=5):
    stems = [f"0919_214639_{int(start[-6:]) + k * step:06d}" for k in range(n)]
    stems = [s for s in stems if label_path(s) and os.path.exists(os.path.join(CAPTURE, s + ".jpg"))]
    frames = [frame(s, width).quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG) for s in stems]
    out = os.path.join(OUT, "labels_walkaround.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(out, len(frames), "frames", round(os.path.getsize(out) / 1e6, 1), "MB")

def hard_cases(width=1600):
    """Pick by the labels themselves: far, cut off, crowded, oblique."""
    stems = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(LABELS, "*", "labels", "*.txt"))]
    stats = {}
    for s in stems:
        gs = gates(s)
        if not gs or not os.path.exists(os.path.join(CAPTURE, s + ".jpg")): continue
        big = max(gs, key=lambda g: g[0][2] * g[0][3])
        vis = sum(v for _, _, v in big[1])
        stats[s] = dict(n=len(gs), h=big[0][3], vis=vis,
                        skew=abs((big[1][0][1] - big[1][1][1])) if big[1][0][2] and big[1][1][2] else 0)
    far = min((s for s in stats if stats[s]["vis"] == 8 and stats[s]["h"] > 0.08), key=lambda s: stats[s]["h"])
    cut = max((s for s in stats if 4 <= stats[s]["vis"] <= 6), key=lambda s: stats[s]["h"])
    crowd = max((s for s in stats if stats[s]["vis"] == 8), key=lambda s: (stats[s]["n"], stats[s]["h"]))
    oblique = max((s for s in stats if stats[s]["vis"] == 8 and stats[s]["h"] > 0.3), key=lambda s: stats[s]["skew"])
    picks = [(far, "far"), (cut, "cut off"), (crowd, "crowded"), (oblique, "oblique")]
    w = width // 2; h = int(w * 9 / 16); gap = 6
    sheet = Image.new("RGB", (2 * w + gap, 2 * h + gap), (13, 17, 23))
    for i, (s, tag) in enumerate(picks):
        im = frame(s, w); im = im.crop((0, 0, w, h))
        sheet.paste(im, ((i % 2) * (w + gap), (i // 2) * (h + gap)))
        print(tag, s, stats[s])
    out = os.path.join(OUT, "labels_hard_cases.jpg"); sheet.save(out, quality=86); print(out)

if __name__ == "__main__":
    walkaround_gif(); hard_cases()

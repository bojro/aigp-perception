"""Draw what is actually in a YOLO-pose dataset, so it can be judged by eye.

Reads the written label files rather than re-deriving anything, so what you
see is exactly what a trainer would consume - including the keypoints marked
unknown, which are drawn hollow so a missing corner is visibly a decision
rather than an oversight.
"""
import argparse, csv, html, subprocess, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SEEN, GUESSED, UNKNOWN = (90, 230, 90), (0, 200, 255), (110, 110, 120)


def draw(image, rows):
    h, w = image.shape[:2]
    for row in rows:
        f = row.split()
        kp = np.array([float(v) for v in f[5:]], float).reshape(-1, 3)
        pts = kp[:, :2] * [w, h]
        vis = kp[:, 2]
        cx, cy, bw, bh = (float(v) for v in f[1:5])
        x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        if not (vis > 0).any():          # box-only: gate asserted, corners not
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 165, 255), 3)
            cv2.putText(image, "box only", (x0 + 6, max(y0 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            continue
        for ring, colour in ((slice(0, 4), (90, 230, 90)), (slice(4, 8), (255, 200, 60))):
            ring_pts = pts[ring]; ring_vis = vis[ring]
            if (ring_vis > 0).all():
                cv2.polylines(image, [ring_pts.astype(np.int32)], True, colour, 3)
        for k, (p, v) in enumerate(zip(pts, vis)):
            q = np.clip(p.astype(int), [14, 14], [w - 14, h - 14])
            if v == 0:
                cv2.circle(image, tuple(q), 9, UNKNOWN, 2)      # hollow = not claimed
            else:
                cv2.circle(image, tuple(q), 9, SEEN if v == 2 else GUESSED, -1)
                cv2.putText(image, str(k), tuple(q + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=240)
    args = ap.parse_args()

    out = args.out or (args.dataset / "preview")
    (out / "img").mkdir(parents=True, exist_ok=True)

    # rank frames by how much the dataset claims about them
    labels = []
    for split in ("train", "valid"):
        labels += sorted((args.dataset / split / "labels").glob("*.txt"))
    picked = []
    for lp in labels:
        rows = [r for r in lp.read_text().split("\n") if r.strip()]
        if not rows:
            continue
        claimed = sum(sum(1 for t in r.split()[7::3] if t != "0") for r in rows)
        picked.append((lp, rows, len(rows), claimed))
    step = max(len(picked) // args.limit, 1)
    picked = picked[::step][:args.limit]

    cards = []
    for lp, rows, ngates, claimed in picked:
        src = args.frames / f"{lp.stem}.jpg"
        image = cv2.imread(str(src))
        if image is None:
            continue
        cv2.imwrite(str(out / "img" / f"{lp.stem}.jpg"),
                    cv2.resize(draw(image, rows), (1280, 720)))
        cards.append((lp.stem, ngates, claimed))

    rows_html = "\n".join(
        f'<figure><img loading="lazy" src="img/{html.escape(s)}.jpg">'
        f'<figcaption><b>{html.escape(s)}</b>'
        f'<span>{n} gate{"s" if n > 1 else ""} &middot; {c} keypoints claimed</span>'
        f'</figcaption></figure>' for s, n, c in cards)
    (out / "index.html").write_text(f"""<!doctype html><meta charset="utf-8">
<title>Gate labels</title><style>
:root{{color-scheme:dark;--bg:#111218;--card:#1b1d26;--ink:#e9eaf2;--dim:#9aa0b5}}
body{{margin:0;padding:26px 16px 64px;background:var(--bg);color:var(--ink);
font:15px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif}}
header{{max-width:1500px;margin:0 auto 22px}} h1{{font-size:21px;margin:0 0 6px}}
p{{margin:0;color:var(--dim)}}
.grid{{max-width:1500px;margin:0 auto;display:grid;gap:18px;
grid-template-columns:repeat(auto-fill,minmax(440px,1fr))}}
figure{{margin:0;background:var(--card);border-radius:12px;overflow:hidden}}
img{{width:100%;display:block;background:#000}}
figcaption{{padding:10px 13px;display:flex;justify-content:space-between;gap:12px;font-size:13px}}
figcaption span{{color:var(--dim);white-space:nowrap}}
@media(max-width:520px){{.grid{{grid-template-columns:1fr}}}}
</style><header><h1>Gate labels as exported</h1>
<p>{len(cards)} frames. Green outline = outer square, amber = opening.
Filled dot = measured, hollow grey = deliberately not claimed.
An amber box with no dots is a gate asserted without corners.</p></header>
<div class="grid">{rows_html}</div>""")
    print(f"{len(cards)} frames -> {out/'index.html'}")


if __name__ == "__main__":
    main()

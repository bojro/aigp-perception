"""Render a clean view of only the auto-accepted labels, plus an index page.

The review overlays draw every candidate in a frame, so an accepted gate sits
among ambers and reds. Here each frame is redrawn with its accepted instances
alone, which is what you want when judging whether the accepted set is good.
"""
import csv, html, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp_perception import paths as P
import aigp_perception.autolabel_gate_pose as A

SRC = P.CAPTURE
out = Path(sys.argv[1])
dest = out / "auto_only"
dest.mkdir(parents=True, exist_ok=True)

wanted = defaultdict(list)
for row in csv.DictReader((out / "report.csv").open()):
    if row["verdict"] == "auto":
        wanted[row["file"]].append(row)

cfg = A.AutolabelConfig()

# Rebuild the same learned gate appearance the tool used, so the verdicts here
# match the ones in report.csv rather than a different run's.
_faces = []
_all = sorted(SRC.glob("*.jpg"))
for _p in _all[:: max(len(_all) // 240, 1)]:
    _img = cv2.imread(str(_p))
    if _img is None:
        continue
    _grey = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for _it in A.gate_candidates(_img, cfg):
        if (_it.verdict == "auto" and _it.anchor == "both" and not _it.clipped
                and _it.support >= cfg.appearance_teacher_support
                and _it.outer_side_px >= cfg.appearance_teacher_side):
            _f = A.rectified_face(_grey, _it.outer)
            if _f is not None:
                _faces.append(_f)
APPEARANCE = A.learn_appearance(_faces)
print(f"appearance from {len(_faces)} faces")

cards = []
for name in sorted(wanted):
    image = cv2.imread(str(SRC / name))
    if image is None:
        continue
    accepted = [i for i in A.gate_candidates(image, cfg, APPEARANCE)
                if i.verdict == "auto"]
    if not accepted:
        continue
    for item in accepted:
        pts = item.points.astype(np.int32)
        cv2.polylines(image, [pts[:4]], True, (90, 230, 90), 3)
        cv2.polylines(image, [pts[4:]], True, (255, 200, 60), 3)
        for index, point in enumerate(pts):
            cv2.circle(image, tuple(point), 9, (255, 0, 255), -1)
            cv2.circle(image, tuple(point), 9, (255, 255, 255), 2)
            cv2.putText(image, str(index), tuple(point + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.imwrite(str(dest / name), cv2.resize(image, (1280, 720)))
    # Single-anchor labels have no ring-agreement number, so rank those by
    # how strongly the image confirms them instead.
    agree = [float(r["residual"]) for r in wanted[name] if r["residual"] != "inf"]
    support = min(float(r["support"]) for r in wanted[name])
    rank = max(agree) if agree else 0.0
    cards.append((name, len(accepted), rank, support,
                  max(float(r["outer_side_px"]) for r in wanted[name])))

cards.sort(key=lambda c: (-c[3], c[2]))
rows = "\n".join(
    f'<figure><img loading="lazy" src="auto_only/{html.escape(n)}" alt="{html.escape(n)}">'
    f'<figcaption><b>{html.escape(n)}</b><span>{k} gate{"s" if k > 1 else ""}'
    f' &middot; {s:.0f}px &middot; confirmed {sup:.2f}</span></figcaption></figure>'
    for n, k, r, sup, s in cards
)
(out / "auto_gallery.html").write_text(f"""<!doctype html>
<meta charset="utf-8"><title>Auto-accepted gate labels</title>
<style>
 :root {{ color-scheme: dark; --bg:#12131a; --card:#1c1e28; --ink:#e9eaf2; --dim:#9aa0b5; }}
 body {{ margin:0; padding:28px 16px 64px; background:var(--bg); color:var(--ink);
        font:15px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif; }}
 header {{ max-width:1500px; margin:0 auto 24px; }}
 h1 {{ font-size:22px; margin:0 0 6px; letter-spacing:-.01em; }}
 p {{ margin:0; color:var(--dim); }}
 .grid {{ max-width:1500px; margin:0 auto; display:grid; gap:18px;
          grid-template-columns:repeat(auto-fill,minmax(440px,1fr)); }}
 figure {{ margin:0; background:var(--card); border-radius:12px; overflow:hidden;
           box-shadow:0 1px 3px rgba(0,0,0,.4); }}
 img {{ width:100%; display:block; background:#000; }}
 figcaption {{ padding:10px 13px; display:flex; justify-content:space-between;
               gap:12px; font-size:13px; }}
 figcaption span {{ color:var(--dim); white-space:nowrap; }}
 @media (max-width:520px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style>
<header>
  <h1>Auto-accepted gate labels</h1>
  <p>{len(cards)} frames &middot; {sum(c[1] for c in cards)} accepted instances.
     Green = outer square (ids 0&ndash;3), amber = opening (ids 4&ndash;7).
     Most strongly confirmed first; scroll down toward the accept threshold.</p>
</header>
<div class="grid">
{rows}
</div>
""")
print(f"{len(cards)} frames, {sum(c[1] for c in cards)} instances -> {out/'auto_gallery.html'}")

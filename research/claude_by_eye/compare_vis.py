"""Render Claude's eyeball estimates against the geometric fit, side by side."""
import cv2, numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aigp_perception import paths as P

SP = P.WORK
SRC = P.CAPTURE
OUT = SP / "compare"; OUT.mkdir(exist_ok=True)
W, H = 1920, 1080

CLAUDE = {
 "0919_214639_000087": [(254,73),(561,97),(551,381),(242,372),(315,147),(472,158),(468,300),(312,295)],
 "0919_214639_000073": [(365,114),(600,117),(600,355),(363,352),(408,171),(553,172),(552,305),(407,304)],
 "0919_220524_000277": [(572,74),(706,96),(688,245),(546,232),(592,120),(668,132),(658,212),(585,202)],
}
CLAUDE_COL, GEOM_COL = (255, 80, 255), (80, 255, 80)   # magenta = me, green = algorithm

def load(stem):
    out = []
    for line in (SP/"full"/"train"/"labels"/f"{stem}.txt").read_text().split("\n"):
        if not line.strip(): continue
        kp = np.array([float(v) for v in line.split()[5:]]).reshape(8, 3)
        out.append((kp[:, :2] * [W, H], kp[:, 2]))
    return out

for stem, guess in CLAUDE.items():
    img = cv2.imread(str(SRC / f"{stem}.jpg"))
    mine = np.array(guess, float) * 2.0
    truth, vis = min(load(stem), key=lambda c: np.linalg.norm(c[0].mean(0) - mine.mean(0)))

    for quad, col in ((truth, GEOM_COL), (mine, CLAUDE_COL)):
        cv2.polylines(img, [quad[:4].astype(np.int32)], True, col, 3)
        cv2.polylines(img, [quad[4:].astype(np.int32)], True, col, 3)

    errs = []
    for i in range(8):
        m, t = mine[i].astype(int), truth[i].astype(int)
        e = float(np.linalg.norm(mine[i] - truth[i])); errs.append(e)
        cv2.line(img, tuple(m), tuple(t), (0, 210, 255), 2)     # the gap itself
        cv2.circle(img, tuple(t), 8, GEOM_COL, -1)
        cv2.circle(img, tuple(m), 8, CLAUDE_COL, -1)
        cv2.putText(img, f"{i}:{e:.0f}px", tuple(m + [12, -12]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    side = np.mean([np.linalg.norm(truth[i] - truth[(i+1) % 4]) for i in range(4)])
    bar = [
        f"{stem}   gate side {side:.0f}px",
        f"mean {np.mean(errs):.1f}px   max {np.max(errs):.1f}px   "
        f"within 20px: {sum(e <= 20 for e in errs)}/8",
    ]
    cv2.rectangle(img, (0, 0), (W, 150), (0, 0, 0), -1)
    for row, text in enumerate(bar):
        cv2.putText(img, text, (24, 48 + row * 46), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (255, 255, 255), 2)
    cv2.putText(img, "MAGENTA = Claude by eye", (W - 780, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, CLAUDE_COL, 2)
    cv2.putText(img, "GREEN = geometric fit", (W - 780, 94),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, GEOM_COL, 2)
    cv2.imwrite(str(OUT / f"{stem}_compare.jpg"), img)
    print(f"{stem}: mean {np.mean(errs):5.1f}px  max {np.max(errs):5.1f}px")
print(f"\nwrote {OUT}")

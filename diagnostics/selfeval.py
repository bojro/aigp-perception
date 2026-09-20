"""How close does Claude-by-eye get to the geometric fit? Estimates were made
from 960x540 views before any label was read; they are doubled to full res."""
import numpy as np
from pathlib import Path

SP = Path(str(Path(__file__).resolve().parents[1] / "work"))
W, H = 1920, 1080

CLAUDE = {  # (outer TL,TR,BR,BL then inner TL,TR,BR,BL) in the 960x540 view
 "0919_214639_000087": [(254,73),(561,97),(551,381),(242,372),(315,147),(472,158),(468,300),(312,295)],
 "0919_214639_000073": [(365,114),(600,117),(600,355),(363,352),(408,171),(553,172),(552,305),(407,304)],
 "0919_220524_000277": [(572,74),(706,96),(688,245),(546,232),(592,120),(668,132),(658,212),(585,202)],
}

def load(stem):
    out = []
    for line in (SP/"full"/"train"/"labels"/f"{stem}.txt").read_text().split("\n"):
        if not line.strip(): continue
        f = line.split()
        kp = np.array([float(v) for v in f[5:]]).reshape(8,3)
        pts = kp[:,:2] * [W,H]
        out.append((pts, kp[:,2]))
    return out

print(f"{'frame':22s} {'pt':>3s} {'claude':>13s} {'geometric':>13s} {'err_px':>7s}")
allerr = []
for stem, guess in CLAUDE.items():
    mine = np.array(guess, float) * 2.0
    cands = load(stem)
    truth, vis = min(cands, key=lambda c: np.linalg.norm(c[0].mean(0) - mine.mean(0)))
    side = np.mean([np.linalg.norm(truth[i]-truth[(i+1)%4]) for i in range(4)])
    errs = []
    for i in range(8):
        if vis[i] == 0: continue
        e = float(np.linalg.norm(mine[i]-truth[i])); errs.append(e); allerr.append(e)
        print(f"{stem:22s} {i:3d} {str(tuple(mine[i].astype(int))):>13s} "
              f"{str(tuple(truth[i].astype(int))):>13s} {e:7.1f}")
    print(f"{'':22s}  -> gate side {side:.0f}px | mean {np.mean(errs):.1f}px "
          f"| max {np.max(errs):.1f}px | as % of side {100*np.mean(errs)/side:.1f}%\n")

a = np.array(allerr)
print(f"OVERALL n={len(a)}  mean={a.mean():.1f}px  median={np.median(a):.1f}px  "
      f"max={a.max():.1f}px  within_20px={100*(a<=20).mean():.0f}%")

"""Cheapest refinement that still pays for itself, at race resolution.

Labelling can afford to be thorough; flight cannot. The question is how much
of the 5x precision gain survives when the work is cut down to what fits
between the detector and the 33 ms frame.
"""
from pathlib import Path
import glob, sys, time, warnings
warnings.filterwarnings("ignore")
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aigp_perception import paths as P
import aigp_perception.autolabel_gate_pose as A

cfg = A.AutolabelConfig()


def refine(field, quad, radius, step, max_stations, passes):
    cur = quad.astype(np.float32).copy()
    offs = np.arange(-radius, radius + step, step, np.float32)
    prox = np.exp(-0.5 * (offs / max(radius, 1e-6)) ** 2)
    for p in range(passes):
        lines = []
        for i in range(4):
            a, b = cur[i], cur[(i + 1) % 4]
            L = float(np.linalg.norm(b - a))
            if L < 12: return None
            d = (b - a) / L; n = np.array([-d[1], d[0]], np.float32)
            ts = np.linspace(0.18, 0.82, int(np.clip(L / 6.0, 6, max_stations)), dtype=np.float32)
            bases = a[None, :] + ts[:, None] * L * d[None, :]
            pts = (bases[:, None, :] + offs[None, :, None] * n[None, None, :]).reshape(-1, 2).astype(np.float32)
            vals = cv2.remap(field, pts[:, 0].reshape(-1, 1), pts[:, 1].reshape(-1, 1),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
            prof = vals.reshape(len(ts), len(offs))
            g = np.abs(np.gradient(prof, axis=1))
            pk = np.argmax(g * prox[None, :], axis=1)
            ok = (pk > 0) & (pk < len(offs) - 1) & (prof.max(1) - prof.min(1) > 14)
            if ok.sum() < 5: return None
            found = bases[ok] + offs[pk[ok]][:, None] * n[None, :]
            vx, vy, x0, y0 = cv2.fitLine(found.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
            lines.append((np.array([x0, y0]), np.array([vx, vy])))
        out = [A._intersect(lines[i], lines[(i + 1) % 4]) for i in range(4)]
        if any(o is None for o in out): return None
        nxt = A.order_corners(np.array(out, np.float32))
        if (np.linalg.norm(nxt - cur, axis=1) > 0.25 * A.mean_side_length(cur)).any():
            return None
        cur = nxt
    return cur


# Reference gates at race resolution, with a network-like error injected.
cases = []
rng = np.random.default_rng(7)
for path in P.frames()[::15]:
    full = cv2.imread(path)
    if full is None: continue
    small = cv2.resize(full, (640, 360))
    field = A.orange_field(small)
    for it in A.gate_candidates(small, cfg):
        if it.verdict != "auto" or it.verified is None or not it.verified.all(): continue
        if it.outer_side_px < 60: continue
        # 3.8% of gate size, the error the trained network actually shows
        noisy = it.inner + rng.normal(0, 0.038 * it.outer_side_px / 1.4, (4, 2)).astype(np.float32)
        cases.append((field, A.order_corners(noisy.astype(np.float32)), it.inner, it.outer_side_px))
print(f"{len(cases)} gates at 640x360\n")

print(f"{'config':40s} {'err before':>11s} {'err after':>10s} {'ms/gate':>9s}")
for name, radius, step, maxst, passes in (
    ("inner ring, 1 pass, 40 st, step .25", 14.0, 0.25, 40, 1),
    ("inner ring, 1 pass, 20 st, step .50", 14.0, 0.50, 20, 1),
    ("inner ring, 1 pass, 12 st, step .50", 14.0, 0.50, 12, 1),
    ("inner ring, 2 pass, 20 st, step .50", 14.0, 0.50, 20, 2),
):
    before, after, times, ok = [], [], [], 0
    for field, noisy, truth, side in cases:
        b = float(np.median(np.linalg.norm(noisy - truth, axis=1)))
        t0 = time.perf_counter()
        r = refine(field, noisy, radius, step, maxst, passes)
        times.append((time.perf_counter() - t0) * 1000)
        if r is None: continue
        ok += 1
        before.append(b); after.append(float(np.median(np.linalg.norm(r - truth, axis=1))))
    print(f"{name:40s} {np.median(before):10.2f}px {np.median(after):9.2f}px "
          f"{np.median(times):8.3f}  (converged {ok}/{len(cases)})")

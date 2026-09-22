"""What does the refinement actually cost, at the resolution the drone flies?

The 10.6 ms measured earlier was per gate, on 1920x1080, with one cv2.remap
call per station - hundreds of Python round trips. The race stream is 640x360
at 30 Hz. This counts the real arithmetic and times a batched version, so the
decision rests on the work rather than on the prototype's overhead.
"""
import sys
from pathlib import Path
import time, warnings
warnings.filterwarnings("ignore")
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aigp_perception import paths as P
import aigp_perception.autolabel_gate_pose as A

cfg = A.AutolabelConfig()
RADIUS, STEP = cfg.edge_refine_radius, 0.25
SAMPLES = int(2 * RADIUS / STEP) + 1

img = cv2.imread(str(P.CAPTURE / "0919_214639_000073.jpg"))
small = cv2.resize(img, (640, 360))          # the stream the drone actually gets
field_s = A.orange_field(small)

# Find a gate in the downscaled frame the way the runtime would.
ref = [it for it in A.gate_candidates(small, cfg) if it.verdict != "rejected"]
quad = A.order_corners(ref[0].outer) if ref else None
side = A.mean_side_length(quad)
print(f"640x360 frame, gate side {side:.0f}px")

stations = [int(np.clip(float(np.linalg.norm(quad[i]-quad[(i+1)%4])) / 6.0, 9, 80))
            for i in range(4)]
per_ring = sum(stations)
print(f"stations per ring {per_ring} | samples per station {SAMPLES}")
print(f"work per gate = 2 rings x 2 passes x {per_ring} stations x {SAMPLES} samples"
      f" = {2*2*per_ring*SAMPLES:,} bilinear taps")


def batched_refine(field, quad, radius=RADIUS, step=STEP):
    """Same maths, one remap for the whole ring instead of one per station."""
    h, w = field.shape[:2]
    offs = np.arange(-radius, radius + step, step, np.float32)
    prox = np.exp(-0.5 * (offs / max(radius * cfg.edge_proximity_frac, 1e-6)) ** 2)
    allpts, meta = [], []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        L = float(np.linalg.norm(b - a))
        if L < 12: return None
        d = (b - a) / L; n = np.array([-d[1], d[0]], np.float32)
        ts = np.linspace(0.18, 0.82, int(np.clip(L / 6.0, 9, 80)), dtype=np.float32)
        bases = a[None, :] + ts[:, None] * L * d[None, :]
        pts = bases[:, None, :] + offs[None, :, None] * n[None, None, :]
        allpts.append(pts.reshape(-1, 2)); meta.append((i, len(ts), bases, n))
    P = np.concatenate(allpts).astype(np.float32)
    vals = cv2.remap(field, P[:, 0].reshape(-1, 1), P[:, 1].reshape(-1, 1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
    lines, off = [], 0
    for i, ns, bases, n in meta:
        prof = vals[off:off + ns * len(offs)].reshape(ns, len(offs)); off += ns * len(offs)
        g = np.abs(np.gradient(prof, axis=1))
        pk = np.argmax(g * prox[None, :], axis=1)
        ok = (pk > 0) & (pk < len(offs) - 1) & (prof.max(1) - prof.min(1) > 14)
        if ok.sum() < 6: return None
        found = bases[ok] + offs[pk[ok]][:, None] * n[None, :]
        vx, vy, x0, y0 = cv2.fitLine(found.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    out = [A._intersect(lines[i], lines[(i + 1) % 4]) for i in range(4)]
    return A.order_corners(np.array(out, np.float32)) if all(o is not None for o in out) else None


for tag, f, q in (("640x360", field_s, quad),):
    for _ in range(3): batched_refine(f, q)
    N = 300
    t = time.perf_counter()
    for _ in range(N): batched_refine(f, q)
    per = (time.perf_counter() - t) / N * 1000
    print(f"\nbatched refine, one ring, {tag}: {per:.3f} ms")
    print(f"  full gate (2 rings x 2 passes)   : {per*4:.2f} ms")
    print(f"  plus LAB conversion of the frame : ", end="")
    t = time.perf_counter()
    for _ in range(N): A.orange_field(small)
    print(f"{(time.perf_counter()-t)/N*1000:.3f} ms (once per frame, all gates share it)")

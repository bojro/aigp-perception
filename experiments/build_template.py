"""Learn the gate's own appearance from the labels we already trust.

Every accepted label carries a homography from the canonical gate face to the
image. Warping the image back through it lands the gate in a fixed square,
whatever the viewpoint. Averaging many of those gives a clean picture of what a
gate actually looks like - the hexagon columns, the AI-GP wordmark, the sponsor
row - built from this capture rather than assumed.

That template is an independent check the edge score cannot give: edges only
say "something changes here", while the template says "this is a gate, the
right way up, at this scale".
"""
from pathlib import Path
import glob, sys
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp_perception import paths as P
import aigp_perception.autolabel_gate_pose as A

SP = str(P.WORK)
SIZE = 192            # canonical face is 2700mm across -> 192px
MARGIN = 0.60         # canonical square spans this fraction of the tile

canonical = (A.CANONICAL_OUTER * SIZE * MARGIN + SIZE / 2).astype(np.float32)
cfg = A.AutolabelConfig()

stack, used = [], 0
for path in P.frames()[::3]:
    img = cv2.imread(path)
    if img is None: continue
    for it in A.gate_candidates(img, cfg):
        # Only the labels with the strongest evidence teach the template.
        if it.verdict != "auto" or it.anchor != "both": continue
        if it.support < 0.88 or it.outer_side_px < 220 or it.clipped: continue
        H, _ = cv2.findHomography(it.outer.astype(np.float32), canonical, 0)
        if H is None: continue
        warped = cv2.warpPerspective(img, H, (SIZE, SIZE))
        grey = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grey = (grey - grey.mean()) / (grey.std() + 1e-6)
        stack.append(grey); used += 1

stack = np.array(stack)
template = np.median(stack, axis=0)          # median: an occluder cannot drag it
template = (template - template.mean()) / (template.std() + 1e-6)
np.save(f"{SP}/gate_template.npy", template)
spread = stack.std(axis=0)
np.save(f"{SP}/gate_template_spread.npy", spread)

print(f"built from {used} high-confidence labels")
print(f"per-pixel disagreement: p50={np.median(spread):.3f} p90={np.percentile(spread,90):.3f}")
vis = cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite(f"{SP}/gate_template.png", cv2.resize(vis, (384, 384), interpolation=cv2.INTER_NEAREST))
sp = cv2.normalize(spread, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite(f"{SP}/gate_spread.png", cv2.resize(sp, (384, 384), interpolation=cv2.INTER_NEAREST))

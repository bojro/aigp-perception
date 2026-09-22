import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp_perception import paths as P
import aigp_perception.autolabel_gate_pose as A

SP = str(P.WORK)
path = str(P.CAPTURE / "0919_214639_000594.jpg")
img = cv2.imread(path); H, W = img.shape[:2]
cfg = A.AutolabelConfig()
mask = A.orange_mask(img, cfg)

padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
cnts, hier = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
cnts = [c - 1 for c in cnts]; hier = hier[0]

print(f"image {W}x{H}  frame_area={W*H}")
tops = sorted([i for i in range(len(cnts)) if hier[i][3] == -1],
              key=lambda i: -cv2.contourArea(cnts[i]))[:4]
for rank, i in enumerate(tops):
    area = cv2.contourArea(cnts[i])
    kids = sorted([(j, cv2.contourArea(cnts[j])) for j in range(len(cnts)) if hier[j][3] == i],
                  key=lambda k: -k[1])[:4]
    inrange = [j for j, a in kids if cfg.minimum_hole_area_fraction*area < a < cfg.maximum_hole_area_fraction*area]
    touches = A._touches_border(cnts[i], W, H)
    fq = A.fit_quadrilateral(cnts[i])
    qh = A.quad_from_hull(cnts[i], img.shape, clipped=touches)
    print(f"\nparent#{rank} area={area:9.0f} ({100*area/(W*H):4.1f}% of frame) touches_border={touches}")
    print(f"   holes(frac): {[round(k[1]/area,3) for k in kids]}   in_range={len(inrange)}")
    print(f"   fit_quadrilateral={'OK' if fq is not None else 'None'}   quad_from_hull={'OK' if qh is not None else 'None'}")

print("\n--- what gate_candidates returns ---")
for item in A.gate_candidates(img, cfg):
    print(f"   verdict={item.verdict:10s} anchor={item.anchor:6s} side={item.outer_side_px:7.0f} "
          f"res={item.residual:.4f} flags={item.flags} reason={item.reason}")

vis = img.copy()
vis[mask > 0] = (0.6*vis[mask > 0] + 0.4*np.array([0,140,255])).astype(np.uint8)
for i in tops:
    cv2.drawContours(vis, cnts, i, (0,255,0), 4)
    for j in range(len(cnts)):
        if hier[j][3] == i and cv2.contourArea(cnts[j]) > 0.01*cv2.contourArea(cnts[i]):
            cv2.drawContours(vis, cnts, j, (0,0,255), 4)
cv2.imwrite(f"{SP}/why594.jpg", cv2.resize(vis, (1100, 619)))

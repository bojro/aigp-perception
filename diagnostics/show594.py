import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp_perception import paths as P
import aigp_perception.autolabel_gate_pose as A
SP = str(P.WORK)
img = cv2.imread(str(P.CAPTURE / "0919_214639_000594.jpg"))
H, W = img.shape[:2]
colour = {"auto": (90,230,90), "review": (0,190,255), "rejected": (60,60,230)}
for item in A.gate_candidates(img, A.AutolabelConfig()):
    c = colour[item.verdict]
    pts = item.points
    cv2.polylines(img, [pts[:4].astype(np.int32)], True, c, 4)
    cv2.polylines(img, [pts[4:].astype(np.int32)], True, c, 3)
    for k, p in enumerate(pts):
        q = np.clip(p.astype(int), [14,14], [W-14, H-14])
        inside = 0 <= p[0] < W and 0 <= p[1] < H
        cv2.circle(img, tuple(q), 11, (255,0,255) if inside else (0,255,255), -1)
        cv2.putText(img, str(k), tuple(q+16), 0, 1.0, (255,255,255), 3)
    cv2.putText(img, f"{item.verdict} res={item.residual:.3f} {'+'.join(item.flags)}",
                tuple(np.clip(pts[0].astype(int)+[0,-18],[10,30],[W-500,H-10])), 0, 1.1, c, 3)
cv2.imwrite(f"{SP}/show594.jpg", cv2.resize(img, (1400, 787)))
print("written")

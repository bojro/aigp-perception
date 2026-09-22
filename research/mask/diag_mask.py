import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aigp_perception import paths as P
from aigp_perception.autolabel_gate_pose import AutolabelConfig, orange_mask
SP = str(P.WORK)
SRC = str(P.CAPTURE)
cfg = AutolabelConfig()
for stem in ("0919_214639_000073", "0919_214639_000087", "0919_220524_000277"):
    img = cv2.imread(f"{SRC}/{stem}.jpg")
    mask = orange_mask(img, cfg)
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hier = hier[0]
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    tops = sorted([i for i in range(len(cnts)) if hier[i][3] == -1],
                  key=lambda i: -cv2.contourArea(cnts[i]))[:5]
    print(f"\n{stem}: {len(cnts)} contours, {sum(1 for i in range(len(cnts)) if hier[i][3]==-1)} parents")
    for rank, i in enumerate(tops):
        a = cv2.contourArea(cnts[i])
        kids = [(j, cv2.contourArea(cnts[j])) for j in range(len(cnts)) if hier[j][3] == i]
        kids = sorted([k for k in kids if k[1] > 0.01*a], key=lambda k: -k[1])
        print(f"  parent#{rank} area={a:9.0f} holes={len(kids)} "
              f"top_hole_frac={(kids[0][1]/a if kids else 0):.3f}  (gate expects 0.309)")
        cv2.drawContours(vis, cnts, i, (0,255,0), 3)
        for j,_ in kids[:3]:
            cv2.drawContours(vis, cnts, j, (0,0,255), 3)
    cv2.imwrite(f"{SP}/mask_{stem}.png", cv2.resize(vis, (960,540)))

import sys
from pathlib import Path
import glob
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aigp_perception import paths as P
from aigp_perception.autolabel_gate_pose import AutolabelConfig, orange_mask, fit_quadrilateral
SRC=str(P.CAPTURE); cfg=AutolabelConfig()
stems=["0919_214639_000102","0919_214639_000210","0919_214639_000390",
       "0919_214639_000474","0919_220524_000054","0919_220524_000282"]
for stem in stems:
    img=cv2.imread(f"{SRC}/{stem}.jpg"); H,W=img.shape[:2]
    mask=orange_mask(img,cfg)
    cnts,hier=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE); hier=hier[0]
    tops=sorted([i for i in range(len(cnts)) if hier[i][3]==-1],
                key=lambda i:-cv2.contourArea(cnts[i]))[:2]
    for i in tops:
        a=cv2.contourArea(cnts[i]); pts=cnts[i].reshape(-1,2)
        on_border=((pts[:,0]<=2)|(pts[:,1]<=2)|(pts[:,0]>=W-3)|(pts[:,1]>=H-3)).mean()
        kids=sorted([(j,cv2.contourArea(cnts[j])) for j in range(len(cnts)) if hier[j][3]==i],
                    key=lambda k:-k[1])[:3]
        q=fit_quadrilateral(cnts[i])
        print(f"{stem} parent area={a:8.0f} border_pts={100*on_border:4.1f}% "
              f"outer_quad={'OK' if q is not None else 'FAIL'} "
              f"holes={[(round(k[1]/a,3)) for k in kids]}")

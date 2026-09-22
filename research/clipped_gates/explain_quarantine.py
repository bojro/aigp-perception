import sys
from pathlib import Path
import os
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aigp_perception import paths as P
import aigp_perception.autolabel_gate_pose as A
SP=str(P.WORK)
os.makedirs(f"{SP}/quar", exist_ok=True)
cfg=A.AutolabelConfig()
names=[l.strip() for l in open("/tmp/q.txt") if l.strip()][:9]
for n in names:
    img=cv2.imread(f"{P.CAPTURE}/{n}"); H,W=img.shape[:2]
    mask=A.orange_mask(img,cfg)
    padded=cv2.copyMakeBorder(mask,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
    cnts,hier=cv2.findContours(padded,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    cnts=[c-1 for c in cnts]; hier=hier[0]
    tops=sorted([i for i in range(len(cnts)) if hier[i][3]==-1],key=lambda i:-cv2.contourArea(cnts[i]))[:2]
    info=[]
    for i in tops:
        a=cv2.contourArea(cnts[i])
        fq=A.fit_quadrilateral(cnts[i]); qh=A.quad_from_hull(cnts[i],img.shape,clipped=True)
        kids=[cv2.contourArea(cnts[j]) for j in range(len(cnts)) if hier[j][3]==i]
        info.append(f"area={100*a/(W*H):.1f}% fq={'Y' if fq is not None else 'n'} qh={'Y' if qh is not None else 'n'} holes={len([k for k in kids if k>0.08*a])}")
    print(f"{n}: {' | '.join(info)}")
    vis=img.copy(); vis[mask>0]=(0.6*vis[mask>0]+0.4*np.array([0,140,255])).astype(np.uint8)
    for i in tops: cv2.drawContours(vis,cnts,i,(0,255,0),5)
    cv2.putText(vis, info[0] if info else "none", (20,60),0,1.4,(255,255,255),3)
    cv2.imwrite(f"{SP}/quar/{n}", cv2.resize(vis,(640,360)))

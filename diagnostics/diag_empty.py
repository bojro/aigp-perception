import sys
from pathlib import Path
import glob
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp_perception.autolabel_gate_pose import AutolabelConfig, orange_mask, gate_candidates
SP=str(Path(__file__).resolve().parents[1] / "work")
SRC="/Users/bojro/Downloads/gate frames"
cfg=AutolabelConfig()
files=sorted(glob.glob(f"{SRC}/*.jpg"))[::6]
empties=[]
for p in files:
    img=cv2.imread(p)
    if not [i for i in gate_candidates(img,cfg) if i.verdict!="rejected"]:
        empties.append(p)
print(f"{len(empties)} empty frames of {len(files)}")
for p in empties[::max(len(empties)//9,1)][:9]:
    img=cv2.imread(p); mask=orange_mask(img,cfg)
    cnts,hier=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    vis=img.copy()
    # orange mask tinted, parents in green, holes in red
    vis[mask>0]=(0.5*vis[mask>0]+0.5*np.array([0,140,255])).astype(np.uint8)
    if hier is not None:
        h=hier[0]
        for i,c in enumerate(cnts):
            if cv2.contourArea(c)<900: continue
            cv2.drawContours(vis,cnts,i,(0,255,0) if h[i][3]==-1 else (0,0,255),3)
    cv2.imwrite(f"{SP}/empty_{p.split('/')[-1]}",cv2.resize(vis,(640,360)))

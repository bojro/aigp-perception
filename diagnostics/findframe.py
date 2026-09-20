"""Locate the frame the user screenshotted, so the failure can be debugged exactly."""
import glob
import cv2, numpy as np

import sys
target = cv2.imread(sys.argv[1])   # a screenshot to locate in the capture
t = cv2.resize(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
t = (t - t.mean()) / (t.std() + 1e-6)

best = []
for path in sorted(glob.glob("/Users/bojro/Downloads/gate frames/*.jpg")):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    c = cv2.resize(img, (160, 90)).astype(np.float32)
    c = (c - c.mean()) / (c.std() + 1e-6)
    best.append((float((t * c).mean()), path))
best.sort(reverse=True)
for score, path in best[:5]:
    print(f"{score:.4f}  {path.split('/')[-1]}")

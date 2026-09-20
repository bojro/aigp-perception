"""Grab N frames once, so both models are judged on identical pictures.

Uses the existing gate-inference camera opener rather than a hard-coded
GStreamer string: the carrier board exposes UYVY on some units and the Argus
pipeline on others, and that module already works out which.

Reads only the camera. Never opens MSP, never arms anything.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/dcl/gate-inference")
import cv2
from inference import open_camera

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
out = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/dcl/gate-compare/frames")
out.mkdir(parents=True, exist_ok=True)
cap = open_camera(dev="/dev/video0", w=1920, h=1080, fps=30)
got = 0
t0 = time.time()
frame = None
while got < n:
    ok, frame = cap.read()
    if not ok:
        print("read failed at", got)
        break
    cv2.imwrite(str(out / f"f{got:05d}.jpg"), frame)
    got += 1
cap.release()
print(f"captured {got} frames in {time.time() - t0:.1f}s -> {out}")
print("size:", frame.shape if got else "none")

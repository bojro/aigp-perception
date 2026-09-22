"""Does training on the hybrid labels beat the model we started from?

Both are scored on the same held-out blocks - contiguous stretches of the walk
that no training frame comes near - so the number means generalisation rather
than memory of a near-duplicate neighbour.
"""
import sys, warnings, shutil
warnings.filterwarnings("ignore")
from pathlib import Path
from ultralytics import YOLO
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aigp_perception import paths as P

SP = P.WORK
DATA = str(SP / "hybrid" / "data_abs.yaml")
mode = sys.argv[1]

if mode == "baseline":
    print("=== BASELINE: the model as trained by the teammate, on our held-out blocks ===")
    m = YOLO(str(P.TEAMMATE_PT))
    r = m.val(data=DATA, imgsz=640, device="mps", verbose=False, plots=False,
              project=str(SP / "runs"), name="baseline", exist_ok=True)
    print(f"BOX  mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}")
    print(f"POSE mAP50={r.pose.map50:.4f}  mAP50-95={r.pose.map:.4f}")
else:
    print("=== FINE-TUNE on hybrid labels ===")
    m = YOLO(str(P.TEAMMATE_PT))
    m.train(data=DATA, epochs=60, patience=15, imgsz=640, batch=16,
            device="mps", workers=4, verbose=False, plots=False,
            project=str(SP / "runs"), name="hybrid_ft", exist_ok=True,
            degrees=15.0, translate=0.30, scale=0.7, perspective=0.001,
            fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=10)
    best = SP / "runs" / "hybrid_ft" / "weights" / "best.pt"
    print(f"\n=== FINE-TUNED, same held-out blocks ===")
    r = YOLO(str(best)).val(data=DATA, imgsz=640, device="mps", verbose=False,
                            plots=False, project=str(SP / "runs"),
                            name="ft_val", exist_ok=True)
    print(f"BOX  mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}")
    print(f"POSE mAP50={r.pose.map50:.4f}  mAP50-95={r.pose.map:.4f}")
    print(f"weights: {best}")

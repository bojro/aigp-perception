"""Train A (hand labels merged) and B (auto labels only) under identical settings.

The two runs share images, split, seed, epochs and augmentation. The only
difference is which labels the 340 human-annotated frames carry. Without B,
what those labels bought stays a guess -- and that is what decides whether more
hand annotation is worth anyone's evening.

Initialised from gate_pose_hybrid_v1.pt rather than cold: it is already trained
on this domain and converges much faster. Augmentation matches the run that
produced that model, so the numbers stay comparable to it as well as to each
other.

flipud stays 0. A vertical flip inverts top/bottom ring identity exactly the
way a wrong flip_idx inverts left/right, and nothing would report it.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "false")
from ultralytics import YOLO  # noqa: E402

ROOT = Path("/workspace/gate")
INIT = "/workspace/gate_pose_hybrid_v1.pt"
EPOCHS = int(os.environ.get("EPOCHS", 250))
SEED = 0

DATA_YAML = """path: {root}
train: images/train
val: images/val
names:
  0: gate
kpt_shape: [8, 3]
flip_idx: [1, 0, 3, 2, 5, 4, 7, 6]
"""


def prepare(tag: str) -> Path:
    """Link the shared images into this dataset so labels sit beside them."""
    root = ROOT / tag
    for split in ("train", "val"):
        img_dir = root / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted((ROOT / "images" / split).glob("*.jpg")):
            dst = img_dir / src.name
            if not dst.exists():
                dst.symlink_to(src)
        n_i = len(list(img_dir.glob("*.jpg")))
        lab_dir = root / "labels" / split
        n_l = len(list(lab_dir.glob("*.txt")))
        gates = sum(
            len([x for x in p.read_text().split("\n") if x.strip()])
            for p in lab_dir.glob("*.txt")
        )
        print(f"  {tag}/{split}: {n_i} images, {n_l} labels, {gates} gates", flush=True)
        assert n_i == n_l, f"{tag}/{split}: {n_i} images vs {n_l} labels"
    (root / "data.yaml").write_text(DATA_YAML.format(root=root))
    return root / "data.yaml"


def run(tag: str):
    data = prepare(tag)
    print(f"\n=== TRAIN {tag}  ({EPOCHS} epochs, seed {SEED}) ===", flush=True)
    t0 = time.time()
    model = YOLO(INIT)
    model.train(
        data=str(data), epochs=EPOCHS, imgsz=640, batch=64, device=0,
        seed=SEED, deterministic=True, workers=8, cache=True, amp=True,
        patience=50, project=str(ROOT / "runs"), name=tag, exist_ok=True,
        plots=True, verbose=False,
        # Matches the run that produced gate_pose_hybrid_v1.
        degrees=15.0, translate=0.30, scale=0.7, perspective=0.001,
        fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=10,
    )
    mins = (time.time() - t0) / 60.0
    best = ROOT / "runs" / tag / "weights" / "best.pt"
    scored = YOLO(str(best))
    r = scored.val(data=str(data), imgsz=640, device=0, verbose=False, plots=False,
                   project=str(ROOT / "runs"), name=f"{tag}_val", exist_ok=True)
    print(
        f"\nRESULT {tag}  box_mAP50={r.box.map50:.4f} box_mAP50-95={r.box.map:.4f} "
        f"pose_mAP50={r.pose.map50:.4f} pose_mAP50-95={r.pose.map:.4f} "
        f"minutes={mins:.1f}",
        flush=True,
    )
    scored.export(format="onnx", imgsz=640, opset=12, simplify=True)
    print(f"EXPORTED {tag} -> {best.with_suffix('.onnx')}", flush=True)
    return r


if __name__ == "__main__":
    for tag in (sys.argv[1:] or ["A_merged", "B_auto"]):
        run(tag)
    print("\nALL_DONE", flush=True)

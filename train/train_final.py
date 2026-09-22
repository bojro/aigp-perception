"""Train on everything, then hand over pictures.

No held-out split. The reason a val set exists is to choose a checkpoint and to
estimate generalisation, and neither survives contact with this data: the
frames are one person walking one lap of one venue, so a held-out block is
unseen but not independent, and the previous val set was 72% auto labels, which
made the metric punish the model for learning from the humans. Spending 67 of
434 hand-labelled frames to compute that number was paying for a bad estimate
with the best data we have.

So: every human label goes into training, auto labels fill the frames no human
touched, epochs are fixed rather than chosen by a metric, and the output is
judged by looking at it.

The val path points at the hand-labelled frames purely so training prints
something recognisable. Those frames are IN the training set. That number is
not a generalisation estimate and must not be quoted as one.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "false")
from ultralytics import YOLO  # noqa: E402

ROOT = Path("/workspace/gate")
INIT = os.environ.get("AIGP_INIT", "/workspace/gate_pose_hybrid_v1.pt")
TAG = os.environ.get("AIGP_TAG", "FINAL_all")
EPOCHS = int(os.environ.get("EPOCHS", 150))

# A dataset built elsewhere, already carrying images/train and labels/train.
# Set it and build() is skipped -- the merge has already happened upstream.
PREBUILT = os.environ.get("AIGP_DATA")

# Augmentation. The defaults are the values that produced gate_pose_hand434,
# so that model stays reproducible from this file; the overrides exist because
# the failure being chased changes.
#
# For close-up gates -- the ring running past the frame edge with only part of
# it visible -- the two levers are scale and translate. Ultralytics samples a
# scale gain in [1-scale, 1+scale], so raising scale reaches further into
# zoomed-in framings, and translate walks the gate toward and over the edge
# instead of holding it near the centre. Both make keypoints leave the frame
# more often, which is the point: the model has to learn that a corner it
# cannot see is not a corner somewhere else.
SCALE = float(os.environ.get("AIGP_SCALE", 0.7))
TRANSLATE = float(os.environ.get("AIGP_TRANSLATE", 0.30))
CLOSE_MOSAIC = int(os.environ.get("AIGP_CLOSE_MOSAIC", 10))

DATA_YAML = """path: {root}
train: images/train
val: images/val
names:
  0: gate
kpt_shape: [8, 3]
flip_idx: [1, 0, 3, 2, 5, 4, 7, 6]
"""


def build():
    """Every frame trains. Hand labels where they exist, ours where they do not."""
    root = ROOT / TAG
    hand = set(json.loads(Path("/workspace/hand_stems434.json").read_text()))
    src434 = ROOT / "A434" / "labels"          # hand-where-available, auto elsewhere
    n_img = n_gate = n_hand = 0
    (root / "images/train").mkdir(parents=True, exist_ok=True)
    (root / "labels/train").mkdir(parents=True, exist_ok=True)
    (root / "images/val").mkdir(parents=True, exist_ok=True)
    (root / "labels/val").mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        for img in sorted((ROOT / "images" / split).glob("*.jpg")):
            lab = src434 / split / f"{img.stem}.txt"
            if not lab.exists():
                continue
            dst = root / "images/train" / img.name
            if not dst.exists():
                dst.symlink_to(img)
            (root / "labels/train" / f"{img.stem}.txt").write_text(lab.read_text())
            n_img += 1
            n_gate += len([x for x in lab.read_text().split("\n") if x.strip()])
            n_hand += img.stem in hand

    # Monitoring only, and these frames are in the training set.
    for img in sorted((root / "images/train").glob("*.jpg")):
        if img.stem in hand:
            v = root / "images/val" / img.name
            if not v.exists():
                v.symlink_to(img.resolve())
            (root / "labels/val" / f"{img.stem}.txt").write_text(
                (root / "labels/train" / f"{img.stem}.txt").read_text())

    print(f"  training on {n_img} frames / {n_gate} gates "
          f"({n_hand} of them human-labelled)", flush=True)
    print(f"  val mirrors {len(list((root/'images/val').glob('*.jpg')))} hand frames "
          f"-- IN TRAIN, monitoring only, not a generalisation estimate", flush=True)
    (root / "data.yaml").write_text(DATA_YAML.format(root=root))
    return root / "data.yaml"


if __name__ == "__main__":
    if PREBUILT:
        root = Path(PREBUILT)
        # Ultralytics resolves `path` literally, and the bundle was built on a
        # different machine.
        (root / "data.yaml").write_text(DATA_YAML.format(root=root))
        data = root / "data.yaml"
        n = len(list((root / "images/train").glob("*.jpg")))
        g = sum(len([x for x in f.read_text().split("\n") if x.strip()])
                for f in (root / "labels/train").glob("*.txt"))
        print(f"  prebuilt: {n} frames / {g} gates from {root}", flush=True)
    else:
        data = build()
    print(f"\n=== TRAIN {TAG}: {EPOCHS} epochs, no early stop ===", flush=True)
    print(f"    scale {SCALE}  translate {TRANSLATE}  close_mosaic {CLOSE_MOSAIC}",
          flush=True)
    t0 = time.time()
    m = YOLO(INIT)
    m.train(data=str(data), epochs=EPOCHS, imgsz=640, batch=64, device=0,
            seed=0, deterministic=True, workers=8, cache=True, amp=True,
            patience=0, project=str(ROOT / "runs"), name=TAG, exist_ok=True,
            plots=True, verbose=False,
            degrees=15.0, translate=TRANSLATE, scale=SCALE, perspective=0.001,
            fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=CLOSE_MOSAIC)
    last = ROOT / "runs" / TAG / "weights" / "last.pt"
    print(f"\nTRAINED {TAG} in {(time.time()-t0)/60:.1f} min -> {last}", flush=True)
    YOLO(str(last)).export(format="onnx", imgsz=640, opset=12, simplify=True)
    print(f"EXPORTED -> {last.with_suffix('.onnx')}", flush=True)
    print("FINAL_DONE", flush=True)

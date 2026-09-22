"""Build a training set from the verified labels, split so it cannot leak.

Frames come half a second apart along a walked path, so neighbours are near
duplicates. A random split scores the model on frames it has effectively
memorised - which is why the existing 0.90 mAP cannot be trusted. Holding out
CONTIGUOUS BLOCKS instead means validation frames are separated from every
training frame by a stretch of walking, so the score reflects generalisation.
"""
import csv, os, shutil, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp_perception import paths as P

SRC = P.CAPTURE
LAB = Path(os.environ.get("AIGP_LABELS",
    os.path.expanduser("~/dev/ai-grand-prix/datasets/autolabel")))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "WORKDIR/ds_group")
BLOCK = 40          # frames per block ~ 20 seconds of walking
HOLD_EVERY = 5      # every 5th block becomes validation

accepted = defaultdict(list)
for row in csv.DictReader((LAB / "report.csv").open()):
    if row["verdict"] == "auto":
        accepted[row["file"]].append(row)

# Recover the written label rows for accepted-only frames.
def label_rows(stem):
    for split in ("train", "valid"):
        p = LAB / split / "labels" / f"{stem}.txt"
        if p.exists():
            return [l for l in p.read_text().split("\n") if l.strip()]
    return []

for split in ("train", "val"):
    for kind in ("images", "labels"):
        (OUT / kind / split).mkdir(parents=True, exist_ok=True)

frames = sorted(accepted)
counts = {"train": 0, "val": 0}
gates = {"train": 0, "val": 0}
for index, name in enumerate(frames):
    split = "val" if (index // BLOCK) % HOLD_EVERY == HOLD_EVERY - 1 else "train"
    rows = label_rows(Path(name).stem)
    if not rows:
        continue
    # Keep only rows whose keypoints are actually claimed - the accepted ones.
    keep = [r for r in rows if any(t != "0" for t in r.split()[5:])]
    if not keep:
        continue
    shutil.copy2(SRC / name, OUT / "images" / split / name)
    (OUT / "labels" / split / f"{Path(name).stem}.txt").write_text("\n".join(keep) + "\n")
    counts[split] += 1
    gates[split] += len(keep)

(OUT / "data.yaml").write_text(f"""path: {OUT}
train: images/train
val: images/val
names:
  0: gate
kpt_shape: [8, 3]
flip_idx: [1, 0, 3, 2, 5, 4, 7, 6]
""")
print(f"train {counts['train']} images / {gates['train']} gates")
print(f"val   {counts['val']} images / {gates['val']} gates")
print(f"blocks of {BLOCK} frames, every {HOLD_EVERY}th block held out -> {OUT}")

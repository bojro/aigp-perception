"""Build datasets A (hand labels merged) and B (auto labels only) from one split.

A and B differ in exactly one way: on the 340 frames a human annotated, A uses
the human labels and B uses ours. Same images, same blocks, same order. Any
difference in the trained models is therefore attributable to the labels and
nothing else, which is the only reason B is worth the GPU time.

On those 340 frames the human labels REPLACE ours rather than joining them.
Scored against the humans our labeller matched 427 of their 737 gates and
proposed another 195 badly placed (the commit that measured it counted our
claims as 595, i.e. 168 unmatched; the 195 here also counts proposals on
frames where we claimed a gate the humans did not); keeping both sets would
feed the model those alongside the truth. Our labels are used only where no human looked.

Two traps, both of which fail silently:

  * A keypoint outside the frame normalises outside [0,1] and Ultralytics
    discards the ENTIRE IMAGE as corrupt -- not the offending point. Off-frame
    points are written v=0 and clamped.
  * Random splits leak. Frames are 0.5 s apart along a walk, so neighbours are
    near-duplicates: the same model reads 0.517 box mAP50-95 on a random split
    and 0.283 on contiguous blocks. Blocks only.
"""
from __future__ import annotations

import json, os, re, shutil, sys, collections
from pathlib import Path

from PIL import Image

# Everything here is data that lives outside the repo (hundreds of MB, all of
# it regenerable or re-exportable), so the locations are environment settings
# rather than constants.
CAPTURE = Path(os.environ.get("AIGP_CAPTURE",
                              os.path.expanduser("~/Downloads/gate frames")))
AUTO    = Path(os.environ.get("AIGP_AUTO_LABELS",
                              os.path.expanduser("~/dev/ai-grand-prix/datasets/hybrid2")))
COCO    = Path(os.environ.get("AIGP_COCO",
                              "coco/train/_annotations.coco.json"))
OUT     = Path(os.environ.get("AIGP_BUNDLE", "bundle"))

BLOCK, HOLD_EVERY = 40, 5          # every 5th block of 40 frames is held out
LONG_SIDE = 640                    # Ultralytics letterboxes to this anyway
RX = re.compile(r"^(.+?)_jpg\.rf\.[0-9A-Za-z]+\.jpg$")

DATA_YAML = """path: {path}
train: images/train
val: images/val
names:
  0: gate
kpt_shape: [8, 3]
flip_idx: [1, 0, 3, 2, 5, 4, 7, 6]
"""


def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def hand_labels() -> dict[str, list[str]]:
    """COCO keypoints -> YOLO pose lines, keyed by capture stem."""
    d = json.load(COCO.open())
    gate_cat = next(c["id"] for c in d["categories"] if c["name"] == "gate")
    by_id = {}
    for im in d["images"]:
        m = RX.match(im["file_name"])
        if m:
            by_id[im["id"]] = (m.group(1), float(im["width"]), float(im["height"]))
    out: dict[str, list[str]] = collections.defaultdict(list)
    off_frame = 0
    for a in d["annotations"]:
        if a["category_id"] != gate_cat or a["image_id"] not in by_id:
            continue
        stem, W, H = by_id[a["image_id"]]
        x, y, w, h = a["bbox"]
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        bw, bh = w / W, h / H
        fields = [f"{clamp01(cx):.6f}", f"{clamp01(cy):.6f}",
                  f"{clamp01(bw):.6f}", f"{clamp01(bh):.6f}"]
        kp = a["keypoints"]
        for i in range(0, len(kp), 3):
            px, py, v = kp[i] / W, kp[i + 1] / H, int(kp[i + 2])
            if not (0.0 <= px <= 1.0 and 0.0 <= py <= 1.0):
                # Outside the frame: Ultralytics would drop the whole image.
                v, off_frame = 0, off_frame + 1
                px, py = clamp01(px), clamp01(py)
            fields += [f"{px:.6f}", f"{py:.6f}", str(v)]
        out[stem].append("0 " + " ".join(fields))
    print(f"  hand: {sum(len(v) for v in out.values())} gates on {len(out)} frames "
          f"({off_frame} off-frame keypoints written v=0)")
    return dict(out)


def auto_labels() -> dict[str, list[str]]:
    out = {}
    for split in ("train", "valid"):
        for p in (AUTO / split / "labels").glob("*.txt"):
            out[p.stem] = [l for l in p.read_text().split("\n") if l.strip()]
    print(f"  auto: {sum(len(v) for v in out.values())} gates on "
          f"{sum(1 for v in out.values() if v)} frames")
    return out


def main() -> int:
    hand, auto = hand_labels(), auto_labels()
    stems = sorted(p.stem for p in CAPTURE.iterdir() if p.suffix.lower() == ".jpg")
    print(f"  capture: {len(stems)} frames")

    train_all = os.environ.get("AIGP_SPLIT") == "all"
    if train_all:
        # Every frame trains. There is then no held-out number and the model is
        # judged by eye on the aircraft instead -- a deliberate trade, taken
        # because the labels are scarce and the failure we are chasing
        # (close-up gates that crop at the frame edge) is easier to see in a
        # live overlay than to capture in a mAP over a handful of frames.
        #
        # A few frames are still copied into val, because Ultralytics needs a
        # validation loader to run at all. They are training frames. The
        # metrics printed against them measure fit and NOTHING about
        # generalisation, and must never be quoted as a result.
        split_of = {s: "train" for s in stems}
    else:
        split_of = {s: ("val" if (i // BLOCK) % HOLD_EVERY == HOLD_EVERY - 1 else "train")
                    for i, s in enumerate(stems)}

    # Does the held-out side actually contain human labels? Without them the
    # evaluation has no ground truth and only self-scored numbers come back.
    if train_all:
        print(f"\n  AIGP_SPLIT=all: every one of {len(stems)} frames trains, "
              f"{len(hand)} of them hand-labelled. No held-out set.")
    else:
        val_hand = [s for s in stems if split_of[s] == "val" and s in hand]
        print(f"\n  val blocks hold {sum(1 for s in stems if split_of[s]=='val')} frames, "
              f"of which {len(val_hand)} are hand-labelled "
              f"({sum(len(hand[s]) for s in val_hand)} gates)")
        if not val_hand:
            raise SystemExit("no hand labels in the held-out blocks: no ground truth")

    resized: dict[str, Path] = {}
    shared = OUT / "images"
    for split in ("train", "val"):
        (shared / split).mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(stems):
        src = CAPTURE / f"{s}.jpg"
        dst = shared / split_of[s] / f"{s}.jpg"
        if not dst.exists():
            try:
                im = Image.open(src).convert("RGB")
            except Exception:  # noqa: BLE001 - a bad frame is skipped, not fatal
                continue
            w, h = im.size
            k = LONG_SIDE / max(h, w)
            im.resize((round(w * k), round(h * k)), Image.LANCZOS).save(
                dst, "JPEG", quality=92)
        resized[s] = dst
        if (i + 1) % 300 == 0:
            print(f"    resized {i+1}/{len(stems)}", flush=True)

    for tag, use_hand in (("A_merged", True), ("B_auto", False)):
        root = OUT / tag
        counts = collections.Counter()
        for split in ("train", "val"):
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
        for s in stems:
            if s not in resized:
                continue
            lines = hand[s] if (use_hand and s in hand) else auto.get(s, [])
            sp = split_of[s]
            link = root / "images" / sp / f"{s}.jpg"
            if not link.exists():
                link.symlink_to(resized[s].resolve())
            (root / "labels" / sp / f"{s}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
            counts[f"{sp}_img"] += 1
            counts[f"{sp}_gate"] += len(lines)
            if use_hand and s in hand:
                counts[f"{sp}_handframe"] += 1
        if train_all:
            # Ultralytics will not start without a validation loader. These are
            # training frames, chosen evenly across the capture so the loader
            # sees the same variety the trainer does. Any metric printed
            # against them is fit, not generalisation.
            usable = [s for s in stems if s in resized]
            step = max(1, len(usable) // 40)
            for s in usable[::step][:40]:
                link = root / "images" / "val" / f"{s}.jpg"
                if not link.exists():
                    link.symlink_to(resized[s].resolve())
                (root / "labels" / "val" / f"{s}.txt").write_text(
                    (root / "labels" / "train" / f"{s}.txt").read_text())
                counts["val_img"] += 1
        (root / "data.yaml").write_text(DATA_YAML.format(path=root))
        print(f"\n  {tag}: train {counts['train_img']} img / {counts['train_gate']} gates"
              f" | val {counts['val_img']} img / {counts['val_gate']} gates"
              f" | hand frames: train {counts['train_handframe']} val {counts['val_handframe']}")
    print(f"\nbundle at {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

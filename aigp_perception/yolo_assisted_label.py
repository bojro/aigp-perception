"""Label gates by letting the detector propose and the geometry dispose.

The two systems fail in opposite directions. The trained detector finds every
gate - on frames it has never seen it missed none of the geometrically
verified ones - but places corners to roughly 4% of gate size. The geometric
engine places corners to well under a pixel, but only where it can trace clean
edges, which leaves the cropped and cluttered gates unverified.

So run the detector for recall, refine each proposal against the image's own
colour edges, and then judge it with exactly the checks the geometric labeller
applies to itself: are the corners backed by two measured sides, is the
opening actually open, does it look like a gate. Measured on this capture that
roughly doubles the number of labels at the same sub-pixel accuracy.

Note the detector's own confidence does not predict which proposals survive
(0.53 for kept versus 0.56 for discarded). The geometry is not re-scoring what
the network already knew; it is supplying information the network does not have.

    python tools/yolo_assisted_label.py "~/Downloads/gate frames" \
        --model ~/Downloads/best.pt --out datasets/hybrid
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp_perception.autolabel_gate_pose import (  # noqa: E402
    AutolabelConfig, DATA_YAML, IMAGE_SUFFIXES, edge_alignment, gate_candidates,
    keypoint_visibility, mean_side_length, normalise_ring_order, opening_is_see_through,
    orange_field, orange_mask, order_corners, refine_edges_subpixel,
    verified_keypoints, yolo_pose_line, GateInstance,
)


def refine_proposal(field, points, config):
    """Pull the detector's eight corners onto the image's own edges."""
    fixed = points.copy().astype(np.float32)
    for ring_start in (0, 4):
        ring = order_corners(fixed[ring_start:ring_start + 4])
        refined = refine_edges_subpixel(
            field, ring, config.edge_refine_radius,
            proximity_frac=config.edge_proximity_frac,
            size_frac=config.edge_size_frac,
        )
        if refined is None:
            return None
        fixed[ring_start:ring_start + 4] = refined
    return normalise_ring_order(fixed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("datasets/hybrid"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--val-block", type=int, default=40,
                        help="frames per held-out block; blocks keep near-duplicate "
                             "neighbours on the same side of the split")
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(args.model.expanduser()))
    config = AutolabelConfig()

    frames = sorted(
        p for p in args.frames.expanduser().iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )[:: max(args.stride, 1)]
    out = args.out.expanduser()
    for split in ("train", "valid"):
        for kind in ("images", "labels"):
            (out / split / kind).mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text(DATA_YAML)

    # Hold out contiguous blocks: neighbours half a second apart are near
    # duplicates, so a random split would score the model on what it memorised.
    hold = max(int(round(1.0 / max(args.val_fraction, 1e-6))), 2) if args.val_fraction > 0 else 0

    rows, kept_geo, kept_net, kept_box, dropped = [], 0, 0, 0, 0
    for index, path in enumerate(frames):
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        field = orange_field(image)
        mask = orange_mask(image, config)

        found = gate_candidates(image, config)
        instances = [item for item in found if item.verdict == "auto"]

        # Keep the box-only gates too. These are the ones the frame has cut so
        # badly that no corner could be confirmed - which is precisely the gate
        # the drone is about to fly through. They claim no keypoints, so they
        # cannot teach a wrong corner; they assert only that a gate is here,
        # and leaving them out teaches the opposite: that a gate filling the
        # view is background. Size is the guard, since a large orange ring that
        # already passed the gate-ness test is not a signage board.
        for item in found:
            if item.verdict != "review" or "box_only" not in item.flags:
                continue
            if item.outer_side_px < config.minimum_outer_side_px * 5:
                continue
            if "unlike_gate" in item.flags:
                continue
            instances.append(item)
            kept_box += 1
        kept_geo += len(instances)

        result = model.predict(image, imgsz=args.imgsz, conf=args.conf,
                               verbose=False, device=args.device)[0]
        proposals = []
        if result.keypoints is not None and len(result.keypoints.xy):
            proposals = [k.cpu().numpy() for k in result.keypoints.xy]

        for pred in proposals:
            side = mean_side_length(order_corners(pred[:4].astype(np.float32)))
            if side < config.minimum_outer_side_px:
                continue
            centre = pred.mean(axis=0)
            if any(np.linalg.norm(centre - it.points.mean(axis=0))
                   < 0.45 * it.outer_side_px for it in instances):
                continue  # the geometry already has this gate
            fixed = refine_proposal(field, pred, config)
            if fixed is None:
                dropped += 1
                continue
            side = mean_side_length(fixed[:4])
            allowed = min(config.maximum_alignment_px,
                          config.relative_alignment * side)
            verified = verified_keypoints(
                field, fixed, allowed, config.minimum_side_stations
            )
            if int(verified.sum()) < config.minimum_confirmed_keypoints:
                dropped += 1
                continue
            # Same spread test the labeller applies to its own output: corners
            # have to be scattered like the gate they describe. A proposal can
            # have every claimed corner land on a real edge and still be wrong,
            # because the edges belong to a pole or to a gate further away.
            claimed = fixed[verified]
            span = max(float(np.ptp(claimed[:, 0])), float(np.ptp(claimed[:, 1])))
            if side > 1 and span / side < config.minimum_corner_spread:
                dropped += 1
                continue
            through, _ = opening_is_see_through(image, mask, fixed[4:])
            if through > config.maximum_opening_orange:
                dropped += 1
                continue
            alignment, _ = edge_alignment(field, fixed[4:])
            item = GateInstance(
                fixed[:4], fixed[4:], math.inf, side, False, anchor="network"
            )
            item.verified = verified
            item.alignment = alignment
            instances.append(item)
            kept_net += 1

        lines = [
            line for line in
            (yolo_pose_line(item, width, height, config) for item in instances)
            if line
        ]
        split = "valid" if hold and (index // args.val_block) % hold == hold - 1 else "train"
        (out / split / "labels" / f"{path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        target = out / split / "images" / path.name
        if not target.exists():
            if args.copy_images:
                shutil.copy2(path, target)
            else:
                target.symlink_to(path.resolve())
        for item in instances:
            rows.append({
                "file": path.name, "source": item.anchor,
                "side_px": round(item.outer_side_px, 1),
                "align_px": round(item.alignment, 2) if math.isfinite(item.alignment) else "inf",
                "confirmed": int(item.verified.sum()) if item.verified is not None else 0,
                "split": split,
            })
        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(frames)}", flush=True)

    with (out / "report.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["file", "source", "side_px", "align_px",
                                "confirmed", "split"])
        writer.writeheader(); writer.writerows(rows)

    print(f"\n{len(frames)} frames -> {len(rows)} gates "
          f"({kept_geo} from geometry, of which {kept_box} box-only; "
          f"{kept_net} recovered from the detector; "
          f"{dropped} proposals discarded).")
    print(f"Dataset under {out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

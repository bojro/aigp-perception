# Trained gate-pose models

The output of this repo's pipeline, kept here so the model and the code that
made it sit together. The flight repo carries the same files under its own
`models/`; this is the copy with the recipe beside it.

## `gate_pose_hand434`

| | |
|---|---|
| Files | `gate_pose_hand434.onnx` (deploy), `gate_pose_hand434.pt` (re-export / fine-tune) |
| Architecture | `yolov8n-pose`, 8 keypoints, imgsz 640 |
| Input / output | `[1,3,640,640]` -> `[1,29,8400]` |
| Training set | 1207 frames / 2289 gates, **434 human-annotated**, rest auto-labelled |
| Initialised from | `gate_pose_hybrid_v1.pt` |
| Schedule | 150 epochs fixed, batch 64, AMP, seed 0 |
| Augmentation | degrees 15, translate 0.30, scale 0.7, perspective 0.001, fliplr 0.5, flipud **0**, mosaic 1.0 |

`flipud` is zero deliberately: a vertical flip inverts top/bottom ring identity
the same way a wrong `flip_idx` inverts left/right, and nothing reports it.

## Reproducing it

    # 1. two datasets differing only in whose labels the hand frames carry
    AIGP_COCO=<roboflow coco export dir> AIGP_BUNDLE=bundle \
        python harnesses/build_ab_datasets.py

    # 2. A/B, to measure what the human labels bought
    python experiments/train_ab.py A_merged B_auto

    # 3. judge against human labels and PnP, not mAP
    python harnesses/eval_gate_pnp.py --weights <best.pt ...> \
        --images <val images> --truth <val labels> --hand-stems <stems.json>

## What is and is not established

**Established**, on 67 held-out human-labelled frames the model never saw
(sibling checkpoint `A_merged`, trained with a clean contiguous-block split):

| | close-gate recall (>30% of frame) |
|---|---|
| `gate_pose_hybrid_v1` | 71.9% (23/32) |
| trained with hand labels | **100% (32/32)** |

Also visible by eye and in no metric here: the incumbent stacks duplicate
nested boxes on a single close gate -- 7 detections for 2 gates in one checked
frame -- which is a flicker source, because whichever duplicate wins NMS
changes frame to frame and the solved pose lurches while the aircraft is still.

**Not established.** `gate_pose_hand434` itself trained on every frame we have,
by choice, so it has **no held-out number**. Its training curve measures fit,
not generalisation, and must not be quoted as a result. Nothing here has flown:
all training data is a person walking around a gate, and the only capture from
the aircraft is 60 frames containing no gate at all.

Corner accuracy claimed before the human labels arrived (0.75 px) was
self-scored -- measured against the gate's own geometric model, which is the
thing the labeller solves for, so a systematic error fits perfectly. Against
humans it is 4.36 px median, p90 9.64. Some of that is a person clicking a
corner at 1920x1080, so true labeller error sits between the two, nearer the
top. Sub-pixel is not a claim we can make.

## `gate_pose_hand497`

Successor to `gate_pose_hand434`, trained 2026-09-21. **Not yet judged** -- it
has neither a held-out number nor a live look, so nothing here is a result.

| | |
|---|---|
| Files | `gate_pose_hand497.onnx` (1200ca4ec0c7ed08...), `gate_pose_hand497.pt` (9700770b546e39f3...) |
| Training set | 1265 frames / 2348 gates, **497 human-annotated** |
| Initialised from | `gate_pose_hybrid_v1.pt` |
| Schedule | 150 epochs fixed, batch 64, AMP, seed 0, no early stop, `last.pt` |
| Augmentation | degrees 15, translate **0.35**, scale **0.85**, perspective 0.001, fliplr 0.5, flipud 0, mosaic 1.0, close_mosaic **15** |

### What changed from hand434

**63 more hand labels**, a strict superset -- 434 shared, 0 dropped. 58 of the
new ones are close-ups, where the gate ring runs past the frame edge and only
part of it is visible. That is the case the model handles worst, and it is why
the augmentation moved: Ultralytics samples a scale gain in `[1-scale, 1+scale]`,
so 0.85 reaches further into zoomed-in framings, and translate walks the gate
over the edge rather than holding it near the centre. Both push keypoints out
of frame more often, which is the point -- the model has to learn that a corner
it cannot see is not a corner somewhere else.

Three things had to be fixed before the data could be trained on at all, each
of which failed silently:

* Roboflow splits three ways and the builder read only `train/`, which would
  have discarded 115 of the 497 human labels. `harnesses/merge_coco_splits.py`
  merges them, reassigning ids -- Roboflow numbers them from zero *inside each
  split*, so appending the lists makes `valid/` annotations point at `train/`
  images without erroring.
* 58 labelled stems were in a different capture directory entirely and resolved
  to nothing. The capture is now assembled from both.
* macOS `tar` wrote 2617 AppleDouble `._*` files during upload. `ls` hides
  them; Python's `glob` does not, so `._*.jpg` would have been scanned as
  corrupt images.

### Reproducing it

    python harnesses/merge_coco_splits.py <roboflow export> -o merged.coco.json
    AIGP_SPLIT=all AIGP_CAPTURE=<frames> AIGP_COCO=merged.coco.json \
        AIGP_BUNDLE=bundle python harnesses/build_ab_datasets.py
    AIGP_DATA=<bundle>/A_merged AIGP_TAG=HAND497 \
        AIGP_SCALE=0.85 AIGP_TRANSLATE=0.35 AIGP_CLOSE_MOSAIC=15 \
        python experiments/train_final.py

`train_final.py`'s defaults are unchanged, so `hand434` remains reproducible
from the same file.

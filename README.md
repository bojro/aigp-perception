# aigp-perception

Offline perception tooling for the AI Grand Prix drone-racing gate task: a
geometric auto-labeller for the eight gate keypoints, a hybrid labeller that
pairs it with a trained detector, and the harnesses used to decide whether any
of it actually works.

Runtime flight code lives in `ai-grand-prix` (`vision/`, `camera_model.py`,
`vision_rx.py`). Nothing here is on the flight path.

## The gate

Square annulus, front face planar: outer boundary 2700 mm, flyable opening
1500 mm, concentric, 260 mm deep (spec 3.7). Eight keypoints, ids 0-3 outer
ring and 4-7 inner ring, each clockwise from top-left, `flip_idx
[1,0,3,2,5,4,7,6]`. Same convention as `ai-grand-prix/models/README.md` and as
the team's Roboflow exports, so labels interchange without translation.

## Why geometry at all

A gate is an orange ring, so a colour mask of it has a hole. The parent contour
is the outer square and the child hole is the opening, which means ring
identity comes from topology rather than from telling eight similar corners
apart - the thing that defeats a vision-language model on this task.

From there it is measurement, not recognition: fit the sides, intersect them
for corners (including corners past the frame edge), refine each side to the
image's own colour step, and fit one homography to all eight points, which is
over-determined and pulls both rings onto a geometrically exact gate.

## What it does not do

It cannot label a gate it cannot trace. Roughly a third of instances end up in
review rather than accepted, and that is the honest outcome - see the hybrid
below for how the detector covers exactly that gap.

## Measured

On 1207 frames of a hand-carried walk-around (1920x1080, 2 fps):

| | |
|---|---|
| accepted gates, geometry alone | 1026 across 816 images |
| accuracy (median offset to the image's own edges) | **0.75 px** |
| accepted gates, hybrid with the detector | **1917** |
| PnP through `vision/yolo_pnp.py` | 86% solve, **0.45 px** reprojection, 0 implausible |

Fine-tuning `yolov8n-pose` on the hybrid labels, scored on held-out contiguous
blocks (see "splits" below):

| | baseline | fine-tuned |
|---|---|---|
| Box mAP50-95 | 0.283 | **0.636** |
| Pose mAP50-95 | 0.760 | **0.854** |
| recall on verified gates | 98% | **100%** |
| keypoint error vs geometry | 11.8 px | **8.3 px** |

Robustness to flight-like motion blur (the training data is walking, not
flying): recall holds at 97-100% out to ~215 deg/s of simulated yaw smear,
keypoint error degrades 9.8 -> 16.3 px. Blur is not the weak point.

## The hybrid, and why it is not just confidence thresholding

The detector missed **0 of 127** geometrically verified gates on frames it had
never seen, but places corners to ~3.8% of gate size. The geometry places them
to 0.2% but only where it can trace them. So the detector proposes and the
geometry disposes: refine each proposal to the image's edges, then apply the
same per-corner checks the labeller applies to itself.

That recovers ~170 extra gates per 200 frames at the same 0.75 px accuracy,
roughly doubling yield. Crucially the detector's own confidence cannot
substitute for the geometric check - recovered proposals score 0.53 and
rejected ones 0.56, no separation at all. The geometry supplies information
the network does not have.

## Splits

Frames come half a second apart along a walked path, so neighbours are near
duplicates and a random train/val split scores the model on what it memorised.
Hold out **contiguous blocks** instead. This is not a detail: the same model
reads 0.517 box mAP50-95 on a random split and 0.283 on blocks.

## The trained model

`models/gate_pose_hand434.onnx` is the current output of this pipeline and the
model the aircraft should fly -- see `models/README.md` for how it was trained,
what is established about it and what is not.

## Layout

    aigp_perception/     the engine - autolabeller, hybrid labeller
    models/              trained weights, ONNX for the Orin
    datasets/            Roboflow merge, A/B dataset builder, label validator, renderer
    train/               the two training scripts (pod)
    eval/                held-out evaluation, model A/B, galleries
    deploy/onnx/         the onnxruntime detector, PnP, dual detector, verify
    deploy/orin/         torch harnesses that ran on the aircraft's computer
    harnesses/           labeller A/B bench and audit crops
    experiments/         the experiments behind the numbers above, results in docstrings
    diagnostics/         one-off scripts used to find specific failures

## Two traps worth knowing

**Off-frame keypoints.** A keypoint outside the frame normalises outside [0,1],
and Ultralytics rejects the label as corrupt - silently discarding the *entire
image*, not the one point. Write them `v=0`; their estimated positions stay in
`report.csv` for PnP.

**`SOLVEPNP_IPPE_SQUARE` object-point order.** It expects a specific corner
ordering and sign convention. Get it wrong and you get reprojection errors of
1e10 px rather than an error. `ai-grand-prix/vision/yolo_pnp.py` does it right.

# aigp-perception

Offline perception tooling for the AI Grand Prix drone-racing gate task: a
geometric auto-labeller for the eight gate keypoints, a hybrid labeller that
pairs it with a trained detector, the training and evaluation scripts that
produced the two shipped `yolov8n-pose` models, and an onnxruntime detector for
the Jetson Orin. Nothing here is on the flight path: the runtime flight code
lives in the team's `ai-grand-prix` repo (`vision/`, `camera_model.py`,
`vision_rx.py`), and this repo's job is to hand it a model and a number for how
much to trust it.

Part of the AI Grand Prix work — see the write-up in
[aigp-sim/paper](https://github.com/bojro/aigp-sim/tree/main/paper).

## The gate

Square annulus, front face planar: outer boundary 2700 mm, flyable opening
1500 mm, concentric, 260 mm deep (spec 3.7). Eight keypoints, ids 0-3 outer
ring and 4-7 inner ring, each clockwise from top-left as seen:

    0 ───────────── 1        out_UL out_UR out_BR out_BL   ids 0 1 2 3
    │  4 ───── 5  │         in_UL  in_UR  in_BR  in_BL    ids 4 5 6 7
    │  │       │  │
    │  7 ───── 6  │         flip_idx [1,0,3,2,5,4,7,6]  (fliplr swaps L/R)
    3 ───────────── 2        flipud 0                      (a vertical flip would
                                                           swap ring identity silently)

Same convention as `ai-grand-prix/models/README.md` and the team's Roboflow
exports, so labels interchange without translation.

## The pipeline, stage by stage

1. **Geometric labels** from a colour mask: `aigp-autolabel <frames> --out work/autolabel`
   (`aigp_perception/autolabel_gate_pose.py`). A gate is an orange ring, so its
   mask has a hole; the parent contour is the outer square and the child hole
   the opening, which gives ring identity from topology rather than from
   telling eight similar corners apart. From there it is measurement: fit the
   sides, intersect them for corners (including corners past the frame edge),
   refine each side to the image's own colour step, and fit one homography to
   all eight points. Frames it cannot trace go to review, about a third.
2. **Hybrid labels**: `aigp-hybrid-label <frames> --model best.pt --out work/hybrid`
   (`aigp_perception/yolo_assisted_label.py`). The detector proposes, the
   geometry disposes: every proposal is refined to the image's edges and put
   through the labeller's own per-corner checks. Roughly doubles yield.
3. **Human labels merged in**: `python datasets/merge_coco_splits.py <roboflow export> -o merged.coco.json`
   (Roboflow numbers ids from zero inside each split, so appending the lists
   silently mis-points annotations) then
   `AIGP_SPLIT=all AIGP_CAPTURE=<frames> AIGP_COCO=merged.coco.json AIGP_BUNDLE=bundle python datasets/build_ab_datasets.py`
   (hand labels replace ours wherever a human looked; contiguous held-out blocks).
4. **Train** on the pod: `AIGP_DATA=<bundle>/A_merged AIGP_TAG=... python train/train_final.py`
   (`train/train_ab.py` for the A/B that measured what the hand labels bought).
   Exact settings for both shipped models are in `models/README.md`.
5. **Judge** against humans and PnP, not mAP:
   `python eval/eval_gate_pnp.py --weights <best.pt> --images <val> --truth <labels> --hand-stems <stems.json>`.
   `eval/compare_models.py` runs two ONNX models side by side.
6. **Deploy**: `python deploy/onnx/verify.py --model models/gate_pose_hand497.onnx --frames <frames>`
   reports the execution provider, latency at 640x360 against a 33 ms budget,
   and how often a detection becomes a pose; non-zero exit if something would
   bite in flight. Flight thresholds: box confidence 0.4, keypoint confidence
   0.25 (`deploy/onnx/gate_detector.py`).

## The numbers that decided things

| | |
|---|---|
| Labeller accuracy | **4.36 px median, p90 9.64** against 340 human-labelled frames. The 0.75 px claimed earlier was self-scored against the gate model the labeller solves for; sub-pixel is not a claim we can make |
| Split leakage | The same model reads **0.517** box mAP50-95 on a random split and **0.283** on contiguous blocks: frames half a second apart are near-duplicates. Hold out blocks |
| Fine-tune on hybrid labels (held-out blocks) | box mAP50-95 0.283 -> **0.636**, pose 0.760 -> **0.854**, recall on verified gates 98 -> 100 %, keypoint error vs geometry 11.8 -> 8.3 px |
| Close-gate recall, hand labels vs none | **71.9 % -> 100 %** (23/32 -> 32/32) on 67 held-out human-labelled frames; close gates (>30 % of frame) are the ones flown through |
| Keypoint threshold | The library default 0.5 was flying, never chosen. At 0.5 the hover runner had four corners **36 %** of the time; at **0.25**, 59 %. The gate was detected in 100 % of frames at every setting |
| Latency on the Orin | **27.4 ms** p50 per model at 640x360, torch, 25 W profile; FP16 changed nothing. One model fits 33 ms; two do not (~61 ms) |
| hand497 vs hand434 at a real gate | 1345 live frames, both at keypoint 0.25: corner flicker **4.2 % vs 6.0 %**, usable fragments **37 vs 78**, longest unbroken run **178 vs 142** frames, policy-ready **59 % vs 56 %** |
| Why the hybrid is not confidence thresholding | The detector missed 0 of 127 verified gates but places corners to ~3.8 % of gate size against the geometry's 0.2 %; recovered proposals score 0.53 and rejected ones 0.56, no separation |

Motion blur is not the weak point: recall holds at 97-100 % out to ~215 deg/s
of simulated yaw smear (keypoint error 9.8 -> 16.3 px).

## What is not established

* **Nothing here has flown.** `deploy/onnx/` has never run on the aircraft;
  `deploy/orin/` ran, on the teammate's torch stack, with the camera at a bench
  and then at one gate.
* **All training data is a person walking around a gate**, at 2 fps, not a
  drone flying one. The only in-flight capture contained no gate.
* **`hand434` and `hand497` have no held-out number.** Both trained on every
  frame we had, by choice. Their training curves measure fit, not
  generalisation. The held-out result above is from a sibling checkpoint
  trained with a clean split.
* The PnP reprojection cap of 2 px in the flight repo rejects 74 % of real
  solves; 8 px keeps them without adding jitter (`deploy/onnx/README.md`).

## Layout

    aigp_perception/     the engine: autolabeller, hybrid labeller, paths
    models/              gate_pose_hand497 (current) and hand434, .pt + .onnx, with the recipe
    datasets/            Roboflow merge, A/B dataset builder, label validator, renderer
    train/               the two training scripts (run on the pod)
    eval/                held-out evaluation vs humans and PnP, model A/B, galleries
    deploy/onnx/         the onnxruntime detector, PnP, dual detector, verify — never flown
    deploy/orin/         torch harnesses that ran on the aircraft's computer
    research/            the one-off experiments behind every constant, with a table

`aigp_perception/paths.py` reads `AIGP_CAPTURE`, `AIGP_TEAMMATE_PT` and
`AIGP_WORK`; the 1207-frame capture and `work/` are not in the repo.
`eval/gallery.py` shells out to ImageMagick's `montage`.
Install with `pip install -e .[detector,deploy]` for ultralytics/torch and
onnxruntime/pillow respectively.

## Two traps worth knowing

**Off-frame keypoints.** A keypoint outside the frame normalises outside [0,1],
and Ultralytics rejects the label as corrupt - silently discarding the *entire
image*, not the one point. Write them `v=0`; their estimated positions stay in
`report.csv` for PnP.

**`SOLVEPNP_IPPE_SQUARE` object-point order.** It expects a specific corner
ordering and sign convention. Get it wrong and you get reprojection errors of
1e10 px rather than an error. `ai-grand-prix/vision/yolo_pnp.py` does it right.

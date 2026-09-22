# Deploying the gate model

Everything here was measured on this laptop over the 1207-frame walk-around
capture, at 640x360, on CPU. **Nothing in this directory has run on the
aircraft.** The one number that cannot be obtained from a laptop is latency on
the Orin, and it is the number most likely to decide whether any of this flies.

## What to deploy

Not the hybrid on its own, which is what we set out to do. The two models fail
in opposite directions and the combination beats either:

| all 1207 frames | detected | posed | reproj px | ms p50 | ms p95 |
|---|---|---|---|---|---|
| hybrid alone | 82% | 65% | 2.48 | 17.9 | 19.2 |
| teammate alone | 97% | 53% | 3.81 | 17.3 | 19.1 |
| **hybrid, else teammate** | **97%** | **68%** | **2.52** | 19.0 | 41.4 |
| both every frame, keep better | 97% | 70% | 2.47 | 42.9 | 46.6 |

Split by how much of the frame the gate fills — close gates being the ones
flown through, and the ones that matter:

| gate > 30% of frame (n=548) | detected | posed | reproj px |
|---|---|---|---|
| hybrid alone | 72% | 41% | 2.44 |
| teammate alone | 100% | **16%** | 5.67 |
| combined | 100% | **46%** | 2.55 |

The teammate's model sees every close gate and returns corners too rough to
solve. Ours poses them two and a half times more often but is near-blind to
28% of them: on the frames it misses its median score is 0.078, so that is a
hole in the model rather than a threshold to lower. The likely cause is ours —
the geometric labeller is weakest on close gates, so they were thin in the
fine-tuning set.

`ms p95` is the catch. `fallback` pays a second inference on the frames where
the primary saw nothing -- 15% by this table's count, about 18% by the tally in
`dual_detector.py`'s docstring, which was taken at a different box threshold -- and on CPU that lands at 41 ms against a 33 ms
budget. On the Orin with TensorRT FP16 it should be far cheaper, but that is a
guess until someone measures it. **Run `verify.py` on the Orin before trusting
the fallback mode**; if it does not fit, `primary_only` still beats either
model alone on posed-frame rate.

## Two fixes worth more than the model choice

**1. `REPROJ_ERR_MAX_PX = 2.0` is too tight** (`vision/yolo_pnp.py:125`). It was
right for the simulator's exact corners. On real frames it rejects 74% of the
gates the model finds, against 33% at 8 px — and what survives the looser cap
is no jumpier: 16.3% of consecutive range steps exceed 1.5 m at 2 px, 11.8% at
8 px. The cap was discarding good solves. This costs more solves than the whole
model swap, and it applies whichever model is flown.

**2. `Gate.aim_point` steers at the wrong place on close gates**
(`inference/inference.py`). It averages whichever corners are visible, so once
the gate is large enough that inner corners leave frame, the target slides onto
the frame instead of the hole. Measured against the PnP centre: about 1% of the
gate's width off while all four inner corners are visible, 11-47% off when
fewer are. The corners leave frame exactly as the drone closes. `gate_detector.Gate`
here prefers the PnP centre and falls back to the average.

## Files

- `gate_detector.py` — the flight-facing API, same surface as
  `inference.GateDetector`, on onnxruntime instead of torch (a Jetson wheel
  rather than a Jetson build). Reports its execution provider; a silent CPU
  fallback is the failure that hides best.
- `gate_pose.py` — standalone eight-keypoint PnP, so the detector can be
  checked without the flight repo. Where `vision/yolo_pnp.py` is importable,
  prefer it; the two agree to within a percent on solve rate.
- `dual_detector.py` — the primary/fallback policy above.
- `verify.py` — run this on the target before anything depends on it. Reports
  provider, latency at the real stream size, and how often a detection becomes
  a pose. Non-zero exit if something would bite in flight.

## Running it

Flight thresholds are box confidence 0.4 and keypoint confidence 0.25
(`gate_detector.py`); every labelling and evaluation tool defaults to 0.25 for
both, which is why their detection counts run higher.

    python deploy/onnx/verify.py --model models/gate_pose_hand497.onnx \
        --frames "~/gate frames" --limit 200

On the flight side the weights are chosen by `YOLO_POSE_MODEL_PATH`
(`config.py:1135`); note its default, `models/gate_pose_v5.pt`, is not in the
repo — `*.pt` is gitignored, so every deployment copies weights in by hand.

## What is still unknown

- Latency on the Orin, for every mode here.
- Whether the close-gate hole closes by relabelling close frames and
  retraining. That is the real fix; the fallback is a patch over it.
- All of this is a person walking a gate, not a drone flying one. Simulated
  motion blur to ~215 deg/s barely dents recall, but no flight footage exists.
- The teammate's model may have trained on frames from this same capture
  (their notebook used it as a test set, and their physical set is from the
  same site). If so its numbers here are optimistic and the gap is wider than
  it looks — but that cuts in our favour, so it does not change the decision.

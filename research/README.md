# research/: the experiments behind the engine's constants

One-off scripts, kept because each one answered a question whose answer is now
a number in `aigp_perception/autolabel_gate_pose.py` or a sentence in a README.
None of them is on any path that matters; all of them were run once, on the
1207-frame walk-around capture, and most read intermediates from `work/`.
**Neither the capture nor `work/` is in the repo.** Set `AIGP_CAPTURE` (and
`AIGP_TEAMMATE_PT` for anything that loads the teammate's YOLO) before running
one, and expect to regenerate `work/` with the engine first.

| script | question it asked | answer | where the answer lives now |
|---|---|---|---|
| `labeller_tuning/labeller_config_ab.py` | Which morphology and sub-pixel settings make the two rings agree with the 1500/2700 geometry? | A heavy close costs ~3 px; the kernels swept here are the ones in `AutolabelConfig` | `close_kernel`, `open_kernel`, `subpix_window` comments in the engine |
| `labeller_tuning/support_eval.py` | Does edge support separate good labels from bad? | Yes: healthy two-ring labels sit near 0.88, single-anchor ones 0.51-0.55 | `edge_support()` and the `cross_check_bonus` of 0.12 |
| `labeller_tuning/support_dist.py` | What does the support distribution look like per verdict and anchor? | The distribution behind the bar above | same |
| `labeller_tuning/residual_tail_drivers.py` | What drives the p90 residual tail: taper, aspect, size, clipping, or the 260 mm depth? | Alignment medians 1.25 px unclipped, 2.75 clipped, 3.25 on the largest gates: measurement difficulty, not a systematic error, so the cap is 4.0 px rather than a tighter one | `maximum_alignment_px = 4.0` |
| `labeller_tuning/worst_residual_crops.py` | What do the nine worst "both"-ring instances actually look like? | Real edges, oblique views; nothing to fix in the fit | (audit only) |
| `clipped_gates/diag_clip.py` | Why does `fit_quadrilateral` fail on gates cut by the frame edge? | The parent contour hugs the border; a polygon fit has no corner to find there | `quad_from_hull()`, which drops border-hugging hull edges and intersects the rest |
| `clipped_gates/explain_quarantine.py` | Why did a frame with a big orange blob yield no label? (reads the stem list from `/tmp/q.txt`) | Per-frame contour/hole/aspect print-out | `quarantine.txt` handling in the engine |
| `clipped_gates/frame594_explain.py`, `frame594_render.py` | Walk one hard frame (`0919_214639_000594`) stage by stage and draw every candidate with its verdict | The case that motivated the hull reader and the mask-based opening | `quad_from_hull()`, `opening_from_mask()` |
| `mask/diag_mask.py` | Are the contour statistics what the geometry predicts? | Hole/parent area 0.309 = (1500/2700)^2, front-on | the 0.10-0.60 hole band in `gate_candidates()` |
| `mask/diag_morph.py` | Does the morphological close eat the aperture of small gates? | At 9x9 yes; 3x3 keeps it | `close_kernel` |
| `mask/diag_empty.py` | Which frames produce nothing at all, and what is in the mask there? | Signage and bar fragments without a hole | the "banner or a sign" discard rule |
| `appearance/tmatch_eval.py` | Does matching a learned gate-face template separate good labels from bad? | Genuine gates score ~0.6, a misread door 0.28 | `appearance_threshold = 0.40`, `appearance_min_coverage = 0.55`, `learn_appearance()` |
| `appearance/tmatch_refine.py` | Can the template *refine* weak labels, judged by edge support it never saw? | Lifts single-anchor labels (.51 -> .54, .55 -> .60) but degrades two-ring ones (.88 -> .72); rejected as a refinement step, kept as a check | commit `a86238e`; joint refinement removed in `1951a3b` |
| `hybrid/crosscheck.py` | How does the teammate's model score against fully verified geometric labels on frames it never saw? | Missed 0 of 127 gates; corners to ~3.8 % of gate size against the geometry's 0.2 % | README, "The hybrid" |
| `hybrid/detector_extras_adjudicated.py` | Are the detector's extra detections real gates the geometry could not trace, or false positives? | Both, and geometry tells them apart where confidence cannot: recovered 0.53 vs rejected 0.56 | `yolo_assisted_label.py`, README |
| `hybrid/hybrid.py` | Does refining the network's corners to the image's edges recover the geometry's precision? | Yes, at the same 0.75 px self-scored accuracy, roughly doubling yield (~170 extra gates per 200 frames) | `aigp-hybrid-label` |
| `hybrid/exp_train.py` | Does fine-tuning on the hybrid labels beat the model we started from, on held-out contiguous blocks? | Box mAP50-95 0.283 -> 0.636, pose 0.760 -> 0.854, recall 98 -> 100 %, keypoint error 11.8 -> 8.3 px; and the same model reads 0.517 on a random split vs 0.283 on blocks | README, "Measured" and "Splits" |
| `hybrid/refinement_cost_640.py` | What does edge refinement cost at 640x360, counting the arithmetic rather than the prototype's overhead? | The prototype's 10.6 ms per gate at 1920x1080 was Python round trips, not work | (decision: refinement stays offline) |
| `hybrid/runtime_cfg.py` | The cheapest refinement that still pays for itself between the detector and a 33 ms frame? | Little of the 5x gain survives once the work is cut to fit; not worth flying | (decision: refinement stays offline) |
| `audit/cropaudit.py`, `audit/borderline.py` | Show accepted and just-below-the-bar labels at 1:1 so few-pixel errors are visible | Set the accept bar by looking, not by the score alone | `minimum_corner_spread = 0.25` (healthy 1.16 vs failures 0.08-0.11), `trust_reach = 0.15` (carried-off-frame error ~6 % at 0.15, 17 % at 0.30, 44 % at 0.50) |
| `audit/findframe.py` | Which capture frame is this screenshot? | NCC on a 160x90 thumbnail | (tool) |
| `claude_by_eye/*` | How close does a vision-language model reading the image get to the geometric fit? | Tens of pixels on the corners and it cannot tell the eight corners apart, which is why the labeller is geometric | README, "Why geometry at all" |

Two more results from scripts that no longer exist because their answer was
absorbed into the engine: the per-ring `suppress` flag cost 131 on-screen outer
corners and was dropped (`3e04894`); reading the opening from the inverted mask
raised close gates with enough points from 49 % to 55 % and gates found from
423 to 480 (same commit). `minimum_side_stations` at 5, 3 and 2 gave 46 %, 48 %
and 54 % of close gates enough points; 3 was chosen.

Deleted rather than moved here, because the engine holds their finished form:
`lines_proto.py` and `lines2.py` (became `quad_from_hull()`),
`build_template.py` (became `learn_appearance()`), `funnel.py` (counted
drop-outs through a stage list that no longer exists), `subset.py` (a
duplicate of `audit/cropaudit.py`) and `mkdataset.py` (superseded by
`datasets/build_ab_datasets.py`). `crop_truth.py` went earlier; its reach
curve is the `trust_reach` row above.

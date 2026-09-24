# deploy/: two stacks, and which one has run on the aircraft

`onnx/` is the candidate flight component: the gate detector on onnxruntime
(`gate_detector.py`), a standalone eight-keypoint PnP (`gate_pose.py`), the
primary/fallback policy over two models (`dual_detector.py`) and the
pre-flight check (`verify.py`). It was measured on a laptop over the
walk-around capture. **It has never run on the aircraft.**

`orin/` is what did run on the aircraft's computer: torch harnesses that
import the teammate's stack from `/home/dcl/gate-inference` (`inference.py`,
`open_camera`) and run both models side by side on the live camera
(`live_compare.py`), grab stills (`capture.py`) and time inference
(`bench_torch.py`). None of them opens the flight controller's serial port.

The latency table in `orin/README.md` -- about 27 ms per model at 640x360 on
the 25 W profile -- is a **torch** number. No onnxruntime or TensorRT number
for the Orin exists in this repo, and it is the one measurement that decides
whether `onnx/dual_detector.py`'s fallback mode fits the 33 ms budget.

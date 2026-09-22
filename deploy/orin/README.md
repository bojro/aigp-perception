# On the Orin

What is actually true of the aircraft's computer, checked on it rather than
assumed, 20 Sep 2026.

## The machine

Orin NX 16GB, JetPack 6 / L4T R36.4.3, Ubuntu 22.04, Python 3.10.12, CUDA 12.6.
Reachable over Wi-Fi as `dcl-orin` (the address is DHCP and moves; it was
`192.168.13.202`). USB networking does **not** come up on macOS — the gadget
enumerates and gives a serial console, but Apple dropped RNDIS, so
`192.168.55.1` has no interface on a Mac. Use Wi-Fi, or a Windows host.

## PyTorch is there, contrary to `pq/flight/README.md`

That file says the Jetson image has no PyTorch and installing one is a
multi-hour job. It was true and is no longer: `/home/dcl/gate-inference/venv`
carries torch 2.11.0, torchvision 0.26.0 and ultralytics 8.4.157, and
`torch.cuda.is_available()` is True.

It only works through the activate script:

    cd ~/gate-inference && source venv/bin/activate

Calling `venv/bin/python` directly fails on `libcupti.so.12`, then
`libcudss.so.0`. Those libraries ship inside the venv's `nvidia/` packages and
under `/usr/local/cuda-12.6/targets/aarch64-linux/lib`, and it is the activate
script that puts them on the loader path. A bare interpreter path looks exactly
like "torch is not installed" and is not.

## Measured latency

Inference only, 640x360 in, GPU, 25W power mode:

| model | p50 | p95 | rate |
|---|---|---|---|
| Etienne's `best.pt` | 27.4 ms | 27.9 ms | 36.5 fps |
| `gate_pose_hybrid_v1.pt` | 27.6 ms | 27.9 ms | 36.2 fps |

FP16 (`--half`) changed nothing measurable. One model fits a 33 ms budget with
about 5 ms to spare, and that is before capture, PnP or control. Both models on
the same frame costs ~61 ms, which is why `live_compare.py` is a viewing tool
and not a flight configuration.

Under both models the GPU sits at `GR3D_FREQ 99%`, RAM at 2.3 of 15.6 GB, and
temperatures at 62-67 °C — saturated but nowhere near throttling. The board is
on the **25W** profile, not MAXN, so there is headroom nobody has taken yet.

## The scripts

- `live_compare.py` — both models on the same live frame, side by side, served
  as MJPEG to a browser. Frames are handed to clients on a condition variable
  with a sequence number: a viewer that polls a shared buffer on a timer sends
  the same picture several times while the next is still being computed, the
  browser queues every copy, and the view slides further behind the longer you
  watch. That was a real bug here and this is the fix.
- `capture.py` — grab N stills once, so an offline comparison judges both
  models on identical pictures.
- `bench_torch.py` — the latency table above. Torch, through the teammate's
  `inference.py`; no onnxruntime number exists for the Orin.

All three read only the camera. None opens MSP, touches `/dev/ttyTHS1`, or
sends anything to the flight controller.

## Running the side-by-side

    scp deploy/orin/live_compare.py orin:~/gate-compare/
    ssh orin
    cd ~/gate-inference && source venv/bin/activate
    cd ~/gate-compare && python live_compare.py --port 8080

Then open `http://<orin>:8080/` and walk the camera at a gate. Stop it with
`pkill -f live_compare`.

## Still unknown

Whether either model is better *on real gates*: the first 60 frames captured
had no gate in them, only hangar ceiling and an AI-GP signage board. Neither
model fired on the board, which is the documented false-positive trap, and
that is the only real-gate evidence collected so far.

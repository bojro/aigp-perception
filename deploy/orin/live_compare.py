"""Both models on the same live frame, side by side in a browser.

Point the camera at a gate and walk around. Left pane is the model Etienne
deployed, right pane is ours: identical frame, identical code, identical
settings, only the weights differ, so anything you see differ is the model.

    python live_compare.py --port 8080
    then open http://<orin>:8080/ from the laptop

Reads only the camera. Never opens MSP, never arms anything.

On smoothness: a viewer that polls a shared buffer on a timer will send the
same picture several times over while the next one is still being computed,
and the browser queues every copy, so the view slides further behind the world
the longer you watch. Frames are handed over on a condition variable instead,
stamped with a sequence number, and a client that asks again before there is
anything new simply waits. Each frame is therefore sent once, to each client,
and a client too slow to keep up misses frames rather than accruing a backlog.
"""
import argparse
import json
import sys
import threading
from pathlib import Path
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "/home/dcl/gate-inference")
import cv2
import numpy as np
from inference import Gate, GateDetector, open_camera

OUTER, INNER = slice(0, 4), slice(4, 8)
C_OUT, C_IN, C_AIM = (0, 200, 255), (0, 255, 90), (255, 80, 255)


class Latest:
    """One slot holding the newest encoded frame, with a sequence number."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpg = None
        self.seq = 0

    def put(self, jpg):
        with self.cond:
            self.jpg = jpg
            self.seq += 1
            self.cond.notify_all()

    def get_after(self, seen, timeout=2.0):
        """Block until a frame newer than ``seen`` exists; None on timeout."""
        with self.cond:
            if self.seq <= seen:
                self.cond.wait(timeout)
            if self.seq <= seen:
                return None, seen
            return self.jpg, self.seq


latest = Latest()
stop = threading.Event()
stats = {"fps": 0.0, "ms_a": 0.0, "ms_b": 0.0, "cap_ms": 0.0, "enc_ms": 0.0}


def draw(frame, gates, title, colour, ms):
    """One pane: rings, aim point, and what the model thinks, over the frame."""
    v = frame.copy()
    for i, g in enumerate(gates):
        primary = i == 0
        for sl, c in ((OUTER, C_OUT), (INNER, C_IN)):
            pts, vis = g.keypoints[sl], g.kpt_visible[sl]
            thick = 2 if primary else 1
            # Per-edge rather than a closed polyline. A polyline needs all four
            # corners, which is precisely the case that fails on a close-up
            # gate -- so the frames worth looking at drew nothing at all. Every
            # edge whose two corners were both detected is drawn; an edge with
            # a missing corner is not drawn, and a missing corner is not
            # marked. Only what the detector actually found appears.
            for i in range(4):
                if not (vis[i] and vis[(i + 1) % 4]):
                    continue
                a, b = pts[i], pts[(i + 1) % 4]
                cv2.line(v, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                         c, thick, cv2.LINE_AA)
            for (x, y), ok in zip(pts, vis):
                if ok:
                    cv2.circle(v, (int(x), int(y)), 4 if primary else 2, c, -1)
        if primary:
            ax, ay = (int(t) for t in g.aim_point)
            cv2.drawMarker(v, (ax, ay), C_AIM, cv2.MARKER_CROSS, 30, 2)
        x1, y1 = int(g.box[0]), int(g.box[1])
        cv2.putText(v, f"{g.conf:.2f}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    bar = 34
    v[:bar] = (v[:bar] * 0.25).astype(np.uint8)
    n_kp = int(gates[0].kpt_visible.sum()) if gates else 0
    cv2.putText(v, f"{title}   {len(gates)} gate(s)  {n_kp}/8 kpts  {ms:4.0f} ms",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2, cv2.LINE_AA)
    return v


def worker(args):
    # Two thresholds, and the second is the one that decides how much of a
    # ring gets drawn. `conf` is the box score, which decides whether a gate is
    # found at all; KPT_CONF_THRES is per-corner, and a corner below it is
    # marked invisible -- so its edges vanish even though the gate was found.
    # It is a class attribute on Gate, set once for both detectors.
    # Per-frame detection statistics, written alongside the stream.
    #
    # One process can hold /dev/video0, so measuring used to mean killing the
    # stream -- which makes the operator work the gate blind, unable to see
    # what the camera sees while walking the range that is being measured.
    # The stream already runs both detectors on every frame; it may as well
    # record what they found.
    stats_fh = open(args.stats_log, "a", buffering=1) if args.stats_log else None
    if stats_fh:
        print(f"[live] logging per-frame detections -> {args.stats_log}", flush=True)

    Gate.KPT_CONF_THRES = args.kpt_conf
    print(f"[live] box conf {args.conf}  keypoint conf {args.kpt_conf}", flush=True)

    a = GateDetector(args.weights_a, imgsz=args.imgsz, conf=args.conf,
                     device="0", half=args.half)
    b = GateDetector(args.weights_b, imgsz=args.imgsz, conf=args.conf,
                     device="0", half=args.half)
    print("[live] both models loaded", flush=True)
    cap = open_camera(dev=args.camera, w=args.cam_w, h=args.cam_h, fps=args.cam_fps)
    print("[live] camera open", flush=True)

    ema = lambda old, new: new if old is None else 0.9 * old + 0.1 * new
    # One process can hold /dev/video0, so if frames are wanted while the
    # stream is up, the stream is what has to write them.
    saver = None
    if args.save_every > 0:
        d = Path(args.save_dir or
                 f"/home/dcl/gate-compare/frames/{time.strftime('%m%d_%H%M%S')}")
        d.mkdir(parents=True, exist_ok=True)
        saver = {"dir": d, "n": 0, "next": time.monotonic()}
        print(f"[live] saving one frame every {args.save_every:g}s -> {d}",
              flush=True)

    ma = mb = mcap = menc = None
    last = time.perf_counter()
    fps = None
    try:
        while not stop.is_set():
            t = time.perf_counter()
            ok, frame = cap.read()
            mcap = ema(mcap, (time.perf_counter() - t) * 1000)
            if not ok:
                time.sleep(0.05)
                continue
            if saver is not None:
                now_s = time.monotonic()
                if now_s >= saver["next"]:
                    # The untouched full-resolution frame, not the annotated
                    # pane: these are training pictures, and overlays drawn on
                    # them would be baked into the label's own evidence.
                    cv2.imwrite(str(saver["dir"] / f"f{saver['n']:05d}.jpg"), frame)
                    saver["n"] += 1
                    saver["next"] = now_s + args.save_every
                    stats.update(saved=saver["n"])

            small = cv2.resize(frame, (args.width, args.height))

            t = time.perf_counter()
            ga = a.detect(small)
            ma = ema(ma, (time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            gb = b.detect(small)
            mb = ema(mb, (time.perf_counter() - t) * 1000)

            if stats_fh is not None:
                def summarise(gates):
                    if not gates:
                        return {"n": 0}
                    g = max(gates, key=lambda q: (q.box[2] - q.box[0]) * (q.box[3] - q.box[1]))
                    x1, y1, x2, y2 = (float(t) for t in g.box)
                    h, w = small.shape[:2]
                    return {
                        "n": len(gates),
                        "conf": round(float(g.conf), 3),
                        # Fraction of the frame the gate fills: the proxy for
                        # how close the aircraft is, which is what the
                        # close-up failure is indexed by.
                        "area": round(max(0.0, x2 - x1) * max(0.0, y2 - y1) / (w * h), 4),
                        "vis": [int(b) for b in g.kpt_visible],
                        "kpt_conf": [round(float(c), 3) for c in g.kpt_conf],
                    }
                stats_fh.write(json.dumps({
                    "t": round(time.monotonic(), 4),
                    "a": summarise(ga), "b": summarise(gb),
                }) + "\n")

            pa = draw(small, ga, args.name_a, (120, 200, 255), ma)
            pb = draw(small, gb, args.name_b, (120, 255, 180), mb)
            sep = np.full((pa.shape[0], 4, 3), 60, np.uint8)

            t = time.perf_counter()
            ok, buf = cv2.imencode(".jpg", np.hstack([pa, sep, pb]),
                                   [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            menc = ema(menc, (time.perf_counter() - t) * 1000)
            if ok:
                latest.put(buf.tobytes())

            now = time.perf_counter()
            fps = ema(fps, 1.0 / max(now - last, 1e-6))
            last = now
            stats.update(fps=fps or 0.0, ms_a=ma or 0.0, ms_b=mb or 0.0,
                         cap_ms=mcap or 0.0, enc_ms=menc or 0.0)
    finally:
        cap.release()
        print("[live] camera released", flush=True)


PAGE = b"""<!doctype html><meta charset=utf-8><title>gate models side by side</title>
<style>body{background:#111;color:#ddd;font:14px system-ui;margin:0;padding:10px;
text-align:center}img{max-width:100%;height:auto}p{color:#888;margin:6px}
b.a{color:#7ec8ff}b.b{color:#78ffb4}#s{color:#777;font:12px ui-monospace,monospace}</style>
<h3 style="margin:4px"><b class=a>left: ETIENNE</b> &nbsp;&middot;&nbsp;
<b class=b>right: HYBRID</b></h3>
<img src="/stream.mjpg">
<p>orange = outer ring &middot; green = inner ring &middot; magenta cross = aim point</p>
<div id=s>&nbsp;</div>
<script>
setInterval(async()=>{try{const r=await fetch('/stats');document.getElementById('s')
.textContent=await r.text()}catch(e){}},1000);
</script>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(PAGE, "text/html")
            return
        if self.path == "/stats":
            s = stats
            body = (f"pipeline {s['fps']:.1f} fps   capture {s['cap_ms']:.0f} ms   "
                    f"etienne {s['ms_a']:.0f} ms   hybrid {s['ms_b']:.0f} ms   "
                    f"encode {s['enc_ms']:.0f} ms").encode()
            self._send(body, "text/plain")
            return
        if self.path != "/stream.mjpg":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()
        seen = 0
        try:
            while not stop.is_set():
                jpg, seen = latest.get_after(seen)
                if jpg is None:
                    continue
                self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights-a", default="/home/dcl/gate-inference/best.pt")
    p.add_argument("--weights-b",
                   default="/home/dcl/gate-compare/models/gate_pose_hybrid_v1.pt")
    p.add_argument("--name-a", default="ETIENNE")
    p.add_argument("--name-b", default="HYBRID")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25,
                   help="box confidence: whether a gate is detected at all")
    p.add_argument("--stats-log", default="",
                   help="append per-frame detection statistics as JSONL")
    p.add_argument("--kpt-conf", type=float, default=0.25,
                   help="per-corner confidence; below this a corner is marked "
                        "invisible and its edges are not drawn")
    p.add_argument("--quality", type=int, default=70)
    p.add_argument("--half", action="store_true")
    p.add_argument("--save-every", type=float, default=0.0,
                   help="seconds between saved stills; 0 disables. Frames 1/30 s "
                        "apart are near-duplicates that cost disk and labelling "
                        "time and leak across a random split, so 0.5 matches the "
                        "2 fps the original walk-around used.")
    p.add_argument("--save-dir", default="")
    p.add_argument("--camera", default="/dev/video0")
    p.add_argument("--cam-w", type=int, default=1920)
    p.add_argument("--cam-h", type=int, default=1080)
    p.add_argument("--cam-fps", type=int, default=30)
    args = p.parse_args()

    t = threading.Thread(target=worker, args=(args,), daemon=True)
    t.start()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    srv.daemon_threads = True
    print(f"[live] serving on port {args.port} (ctrl-c to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        srv.shutdown()
        t.join(timeout=5)


if __name__ == "__main__":
    main()

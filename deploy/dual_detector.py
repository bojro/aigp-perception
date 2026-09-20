"""Two models, because each fails where the other holds.

Measured over 1207 real frames, at 640x360, gates grouped by how much of the
frame they fill:

                        detected   posed   reproj px
    close   ours            72%     41%       2.44
            teammate       100%     16%       5.67
            both           100%     46%       2.55
    all     ours             82%     65%       2.48
            teammate         97%     53%       3.81
            both             97%     68%       2.52

The two failures are opposite. Ours was fine-tuned on labels from the
geometric labeller, which is weakest on close gates, and inherited that as a
hole: on the frames it misses its median score is 0.078, so this is not a
threshold to lower but something it does not see. The teammate's model sees
every close gate and returns corners too rough to solve -- 16% of them pose,
against 41% of the ones ours will admit to.

So: ask ours first and keep its answer, because where it fires its geometry is
worth about a third less reprojection error. Ask theirs only when ours came
back with nothing. That second call is paid on about 18% of frames rather than
all of them, which matters when the budget is one frame at 30 fps.

Running both on every frame and keeping whichever poses better is slightly
stronger again (70% against 68% overall, 46% against 44% close) and costs a
full second inference every frame. ``always`` selects it, for when the
hardware turns out to have the headroom.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from gate_detector import Gate, GateDetector


class DualGateDetector:
    """Primary detector with a second opinion, on the frames that need one."""

    def __init__(self, primary: str | Path, secondary: str | Path,
                 mode: str = "fallback", pose_solver: Optional[Callable] = None,
                 **kwargs):
        if mode not in ("fallback", "always", "primary_only"):
            raise ValueError("mode must be fallback, always or primary_only")
        self.mode = mode
        self.primary = GateDetector(primary, pose_solver=pose_solver, **kwargs)
        self.secondary = (
            None if mode == "primary_only"
            else GateDetector(secondary, pose_solver=pose_solver, **kwargs)
        )
        # Counted so a flight log can show how often the fallback carried the
        # frame; if that number is small the second model can come out.
        self.n_frames = 0
        self.n_secondary = 0
        self.n_secondary_won = 0

    def detect(self, frame: np.ndarray) -> list[Gate]:
        self.n_frames += 1
        first = self.primary.detect(frame)
        if self.secondary is None:
            return first
        if self.mode == "fallback" and first:
            return first

        second = self.secondary.detect(frame)
        if self.mode == "fallback":
            # Primary saw nothing at all, so anything is an improvement.
            if second:
                self.n_secondary += 1
                self.n_secondary_won += 1
            return second

        # mode == "always": keep the side that actually produced a pose, and
        # on a tie the one whose pose reprojects tighter.
        self.n_secondary += 1
        best = _best(first)
        rival = _best(second)
        if rival is None:
            return first
        if best is None:
            self.n_secondary_won += 1
            return second
        if (rival.pose is not None and
                (best.pose is None or
                 rival.pose.reproj_err_px < best.pose.reproj_err_px)):
            self.n_secondary_won += 1
            return second
        return first

    @property
    def provider(self) -> str:
        return self.primary.provider

    @property
    def on_cpu(self) -> bool:
        return self.primary.on_cpu

    def summary(self) -> str:
        if not self.n_frames:
            return "no frames yet"
        return (f"{self.n_frames} frames, second model run on "
                f"{100.0 * self.n_secondary / self.n_frames:.0f}%, "
                f"preferred on {100.0 * self.n_secondary_won / self.n_frames:.0f}%")


def _best(gates: list[Gate]) -> Optional[Gate]:
    return gates[0] if gates else None

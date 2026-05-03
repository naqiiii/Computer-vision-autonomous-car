import time
import math
from config import Config

class SpeedEstimator:
    def __init__(self):
        self.prev_positions = {}
        self.smoothed_speeds = {}
        self.prev_time = time.perf_counter()   # higher resolution than time.time()

    def estimate(self, tracks):
        now = time.perf_counter()
        dt = now - self.prev_time
        self.prev_time = now
        if dt <= 0:
            return self.smoothed_speeds

        speeds = {}
        for track in tracks:
            tid = track["id"]
            x1, y1, x2, y2 = track["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if tid in self.prev_positions:
                px, py = self.prev_positions[tid]
                raw_speed = math.hypot(cx - px, cy - py) / dt

                # Exponential moving average to smooth jitter
                prev = self.smoothed_speeds.get(tid, raw_speed)
                self.smoothed_speeds[tid] = (
                    Config.SPEED_ALPHA * raw_speed
                    + (1 - Config.SPEED_ALPHA) * prev
                )
                speeds[tid] = self.smoothed_speeds[tid]
            else:
                speeds[tid] = 0.0

            self.prev_positions[tid] = (cx, cy)

        # Clean up stale IDs
        active_ids = {t["id"] for t in tracks}
        self.prev_positions  = {k: v for k, v in self.prev_positions.items()  if k in active_ids}
        self.smoothed_speeds = {k: v for k, v in self.smoothed_speeds.items() if k in active_ids}

        return speeds
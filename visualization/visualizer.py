import cv2
import numpy as np

ACTION_COLORS = {
    "MOVE": (0, 200, 0),
    "WARN": (0, 165, 255),
    "STOP": (0, 0, 220),
}
CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

class Visualizer:
    def draw(self, frame, tracks, speeds, action_result, lanes=None, scale=1.0, fps=0.0):
        """
        frame        — original full-resolution frame (for display)
        tracks       — already scaled back to full-res coordinates
        scale        — inference_w / original_w used to scale lane coords back up
        fps          — current FPS to overlay
        """
        action, reasons = action_result
        h, w = frame.shape[:2]

        # ── Lane overlay ──────────────────────────────────────────────
        # Lane coords come from the small inference frame — scale them up
        if lanes:
            inv = 1.0 / scale if scale > 0 else 1.0   # scale UP = divide by scale
            for side in ("left", "right"):
                line = lanes.get(side)
                if line:
                    x1, y1, x2, y2 = line
                    x1 = int(x1 * inv); y1 = int(y1 * inv)
                    x2 = int(x2 * inv); y2 = int(y2 * inv)
                    # Clamp to frame bounds
                    x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
                    y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 200, 0), 3, cv2.LINE_AA)

        # ── Tracks ────────────────────────────────────────────────────
        # Scale font size relative to frame width so it looks right on any resolution
        font_scale = max(0.35, w / 1920)
        font       = cv2.FONT_HERSHEY_SIMPLEX

        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]

            # Clamp bbox to frame bounds
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w-1, x2); y2 = min(h-1, y2)

            tid   = track["id"]
            cls   = CLASS_NAMES.get(track.get("class"), "obj")
            spd   = speeds.get(tid, 0)
            color = ACTION_COLORS.get(action, (0, 200, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{cls}#{tid}  {int(spd)}px/s"
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

            # Keep label box inside frame top
            label_y = max(y1, th + 6)
            cv2.rectangle(frame,
                          (x1, label_y - th - baseline - 4),
                          (x1 + tw + 6, label_y),
                          color, -1)
            cv2.putText(frame, label,
                        (x1 + 3, label_y - baseline - 2),
                        font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Action banner ─────────────────────────────────────────────
        banner_h  = max(44, int(h * 0.055))
        banner_w  = max(200, int(w * 0.22))
        bfont     = max(0.7, w / 1280)
        color     = ACTION_COLORS[action]

        cv2.rectangle(frame, (20, 14), (20 + banner_w, 14 + banner_h), color, -1)
        # Semi-transparent inner shadow for readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 14), (20 + banner_w, 14 + banner_h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.rectangle(frame, (20, 14), (20 + banner_w, 14 + banner_h), color, -1)

        cv2.putText(frame, f"  {action}",
                    (28, 14 + int(banner_h * 0.72)),
                    font, bfont, (255, 255, 255), 2, cv2.LINE_AA)

        # ── Reason string ─────────────────────────────────────────────
        if reasons:
            reason_y = 14 + banner_h + 18
            cv2.putText(frame, reasons[0],
                        (20, reason_y),
                        font, max(0.35, w / 2560),
                        (200, 200, 200), 1, cv2.LINE_AA)

        # ── FPS counter ───────────────────────────────────────────────
        fps_label = f"FPS: {fps:.1f}"
        (fw, fh), _ = cv2.getTextSize(fps_label, font, font_scale, 1)
        cv2.putText(frame, fps_label,
                    (w - fw - 12, fh + 12),
                    font, font_scale,
                    (200, 200, 200), 1, cv2.LINE_AA)

        return frame
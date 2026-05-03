from config import Config

CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

class DecisionEngine:
    # decision/decision_engine.py
 def decide(self, tracks, depth_map, speeds=None):
    action = "MOVE"
    reasons = []
    h, w = depth_map.shape[:2]

    for track in tracks:
        x1, y1, x2, y2 = track["bbox"]
        cy = max(0, min(int(y2 - (y2 - y1) * 0.1), h - 1))
        cx = max(0, min(int((x1 + x2) / 2), w - 1))
        depth = float(depth_map[cy, cx])

        tid     = track["id"]
        cls_name = CLASS_NAMES.get(track.get("class"), "object")
        speed   = speeds.get(tid, 0) if speeds else 0

        # Dynamic warn threshold — fast objects warn earlier
        dynamic_warn = Config.DEPTH_WARN + min(speed * 0.05, 20)

        if depth < Config.DEPTH_STOP:
            action = "STOP"
            reasons.append(f"{cls_name}#{tid} critical (depth={depth:.0f})")
            break
        elif depth < dynamic_warn and action != "STOP":
            action = "WARN"
            reasons.append(
                f"{cls_name}#{tid} approaching "
                f"(depth={depth:.0f}, speed={speed:.0f}px/s)"
            )

    return action, reasons
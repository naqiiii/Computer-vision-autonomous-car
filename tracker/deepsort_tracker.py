# tracker/deepsort_tracker.py
from deep_sort_realtime.deepsort_tracker import DeepSort
from config import Config

class Tracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=Config.MAX_AGE,
            n_init=3,
            nms_max_overlap=0.7,
            embedder="mobilenet",
            half=Config.DEVICE == "cuda",        # FP16 only on GPU
            embedder_gpu=Config.DEVICE == "cuda", # GPU embedder only if CUDA
        )

    def update(self, detections, frame):
        formatted = []
        for x1, y1, x2, y2, conf, cls in detections:
            formatted.append(([x1, y1, x2 - x1, y2 - y1], conf, int(cls)))

        tracks = self.tracker.update_tracks(formatted, frame=frame)
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            l, t, r, b = track.to_ltrb()
            results.append({
                "id":    track.track_id,
                "bbox":  [int(l), int(t), int(r), int(b)],
                "class": track.det_class,
            })
        return results
from ultralytics import YOLO
import torch
from config import Config

class YOLODetector:
    def __init__(self):
        self.model = YOLO(Config.YOLO_MODEL)
        self.model.to(Config.DEVICE)

    def detect(self, frame):
        results = self.model(
            frame,
            conf=Config.YOLO_CONF,
            classes=Config.YOLO_CLASSES,
            verbose=False
        )[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            detections.append([x1, y1, x2, y2, conf, cls])
        return detections
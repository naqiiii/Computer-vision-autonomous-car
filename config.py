# config.py
import torch

class Config:
    # Auto-selects GPU if available, falls back to CPU silently
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    YOLO_MODEL   = "yolov8n.pt"
    YOLO_CONF    = 0.4
    YOLO_CLASSES = [0, 1, 2, 3, 5, 7]

    # Use lighter MiDaS on CPU to keep it usable
    MIDAS_MODEL  = "DPT_Hybrid" if torch.cuda.is_available() else "MiDaS_small"
    DEPTH_EVERY_N_FRAMES = 4 
    DEPTH_STOP   = 30
    DEPTH_WARN   = 60
    MAX_AGE      = 30
    SPEED_ALPHA  = 0.3
    LANE_ROI_TOP = 0.55
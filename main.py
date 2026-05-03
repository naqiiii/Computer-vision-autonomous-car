import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="timm")

import cv2
import time
import os
from detector.yolo_detector    import YOLODetector
from tracker.deepsort_tracker  import Tracker
from depth.midas_depth         import DepthEstimator
from speed.speed_estimator     import SpeedEstimator
from decision.decision_engine  import DecisionEngine
from visualization.visualizer  import Visualizer

# ── Video path ─────────────────────────────────────────────────────
VIDEO_PATH = r"C:\Users\DrugTarget\Desktop\CV_project\videos\sample2.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Could not open video → {VIDEO_PATH}")
    print(f"File exists: {os.path.exists(VIDEO_PATH)}")
    exit(1)

print(f"✓ Video loaded  : {VIDEO_PATH}")
print(f"  Resolution    : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"  FPS           : {cap.get(cv2.CAP_PROP_FPS):.1f}")
print(f"  Total frames  : {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

# ── Models ─────────────────────────────────────────────────────────
print("Loading models...")
detector        = YOLODetector();  print("✓ YOLO loaded")
tracker         = Tracker();       print("✓ DeepSort loaded")
depth_est       = DepthEstimator(); print("✓ MiDaS loaded")
speed_est       = SpeedEstimator()
decision_engine = DecisionEngine()
visualizer      = Visualizer()
print("All models ready — starting pipeline\n")

INFERENCE_W = 640
fps_time    = time.perf_counter()
fps         = 0.0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video finished.")
        break

    scale = INFERENCE_W / frame.shape[1]
    inference_frame = cv2.resize(
        frame,
        (INFERENCE_W, int(frame.shape[0] * scale))
    )

    detections = detector.detect(inference_frame)
    tracks     = tracker.update(detections, inference_frame)
    depth_map  = depth_est.get_depth(inference_frame)
    speeds     = speed_est.estimate(tracks)
    action     = decision_engine.decide(tracks, depth_map, speeds)

    # Scale bboxes back to full-res before visualizer
    display_tracks = [
        {**t, "bbox": [
            int(t["bbox"][0] / scale), int(t["bbox"][1] / scale),
            int(t["bbox"][2] / scale), int(t["bbox"][3] / scale),
        ]}
        for t in tracks
    ]

    # FPS
    now      = time.perf_counter()
    fps      = 0.9 * fps + 0.1 / max(now - fps_time, 1e-6)
    fps_time = now

    output = visualizer.draw(
        frame,
        display_tracks,
        speeds,
        action,
        lanes=None,      # lane detector not used
        scale=scale,
        fps=fps
    )

    cv2.imshow("Output", output)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
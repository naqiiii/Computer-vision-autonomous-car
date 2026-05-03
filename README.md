# Real-Time Autonomous Driving Perception System

A modular, GPU-accelerated computer vision pipeline for real-time object detection, multi-object tracking, monocular depth estimation, speed estimation, and intelligent driving decisions — all rendered live on video.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=flat-square&logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-6236FF?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-27338e?style=flat-square&logo=opencv)

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Usage](#usage)
- [Module Breakdown](#module-breakdown)
- [Performance Optimisations](#performance-optimisations)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

---

## Overview

This project implements a full real-time perception stack for autonomous driving scenarios. Given a video input the system simultaneously detects road objects (cars, pedestrians, trucks, buses, motorcycles, bicycles), tracks them across frames with persistent IDs, estimates per-pixel scene depth, computes smoothed per-object pixel speed, and outputs a driving decision — **MOVE**, **WARN**, or **STOP** — all rendered in a live annotated output window.

The pipeline is fully modular — every component is an independent class that can be swapped, upgraded, or tested in isolation.

---

## Pipeline Architecture

```
Video Frame (full resolution)
        │
        ▼
  Resize to 640px ── inference frame
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  YOLO Detector                     MiDaS Depth
  YOLOv8m · GPU                     DPT_Hybrid · GPU
  Filtered classes                  Inverted · normalised 0–100
        │                                  │
        ▼                                  │
  DeepSort Tracker                         │
  FP16 embedder · n_init=3                │
        │                                  │
        ▼                                  │
  Speed Estimator                          │
  EMA smoothed · perf_counter             │
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
               Decision Engine
           MOVE / WARN / STOP
           Bottom-centre depth sampling
           Speed-aware dynamic threshold
                       │
                       ▼
                  Visualizer
        Scaled back to full resolution
        Colour-coded boxes · FPS · banner
```

---

## Features

- **Object Detection** — YOLOv8m with class-filtered inference (person, bicycle, car, motorcycle, bus, truck), `verbose=False` to suppress console spam
- **Multi-Object Tracking** — DeepSort with MobileNet embedder, FP16 on GPU, `n_init=3` to suppress ghost tracks, object class forwarded through the full pipeline
- **Monocular Depth Estimation** — Intel MiDaS `DPT_Hybrid`, depth inverted and normalised to 0–100 (0 = close, 100 = far) for stable threshold behaviour
- **Speed Estimation** — Pixel displacement with EMA smoothing, high-resolution `perf_counter` timing, automatic stale-track cleanup
- **Decision Engine** — Three-state output (MOVE / WARN / STOP), speed-aware dynamic warn threshold, bottom-centre bbox depth sampling, human-readable reason string
- **Live Visualiser** — Colour-coded bounding boxes, filled label backgrounds, resolution-adaptive font sizing, FPS counter, action banner, reason overlay
- **Performance** — All inference on 640px frames, display at full resolution, bbox coordinates scaled back automatically
- **Auto Device Detection** — Falls back to CPU if CUDA unavailable, auto-switches to `MiDaS_small` on CPU

---

## Project Structure

```
CV_project/
│
├── main.py                         # Entry point
├── config.py                       # All thresholds and model names
├── requirements.txt
│
├── videos/      
│
├── detector/
│   └── yolo_detector.py
│
├── tracker/
│   └── deepsort_tracker.py
│
├── depth/
│   └── midas_depth.py
│
├── speed/
│   └── speed_estimator.py
│
├── decision/
│   └── decision_engine.py
│
└── visualization/
    └── visualizer.py
```

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1060 6 GB | NVIDIA RTX 3060+ |
| RAM | 8 GB | 16 GB |
| CUDA | 11.8 | 12.1 or 12.8 |

CPU-only mode is supported but runs at 1–3 FPS for depth — not suitable for real-time use.

### Software

- Python 3.9+
- CUDA Toolkit 11.8 or 12.x
- NVIDIA Driver 520+

---

## Installation

### Clone the repository

```bash
git clone[ https://github.com/naqiiii/Computer-vision-car.git]
cd CV_project
```

### Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Output window elements

| Element | Description |
|---------|-------------|
| Bounding boxes | Green = MOVE, orange = WARN, red = STOP |
| Label per box | `class#id  speed px/s` |
| Action banner | Top-left — current driving decision |
| Reason string | Below banner — which object triggered the decision |
| FPS counter | Top-right corner |

### Using a webcam instead of a video file

```python
# In main.py, replace the VIDEO_PATH line with:
cap = cv2.VideoCapture(0)   # 0 = default webcam
```

---

## Module Breakdown

### `detector/yolo_detector.py`

Wraps Ultralytics YOLOv8m. Filters to road-relevant classes via `YOLO_CLASSES`, suppresses console output with `verbose=False`. Model moved to GPU at init. Returns `[x1, y1, x2, y2, conf, cls]` per detection.

### `tracker/deepsort_tracker.py`

Wraps `deep_sort_realtime`. MobileNet appearance embedder with FP16 on GPU. `n_init=3` prevents ghost tracks from flashing on screen before confirmation. Detection class is forwarded so downstream modules know object type.

### `depth/midas_depth.py`

Loads Intel MiDaS `DPT_Hybrid` (or `MiDaS_small` on CPU). Uses `@torch.inference_mode()` for maximum throughput. Depth is **inverted** before normalising — raw MiDaS output is high for close objects and low for far objects, which is opposite to intuition and caused far cars to incorrectly trigger STOP. After inversion and normalisation: 0 = close, 100 = far.

> MiDaS produces relative depth — values are consistent within a frame but not comparable in absolute metres across frames.

### `speed/speed_estimator.py`

Computes per-track pixel displacement between frames divided by elapsed time (`time.perf_counter()` for sub-millisecond precision). EMA smoothing (`SPEED_ALPHA`) removes bbox jitter from the tracker. Stale entries are cleaned up automatically each frame.

### `decision/decision_engine.py`

Samples depth at the **bottom-centre** of each bounding box — where the object contacts the road, far more accurate than the geometric centre which can land on the sky or background. Dynamic warn threshold: fast-moving objects trigger WARN earlier than stationary ones. Returns action string and reason list.

### `visualization/visualizer.py`

Draws on the full-resolution display frame. Lane coordinates (detected on the small inference frame) are multiplied by `1/scale` before drawing. Bbox coordinates are clamped to frame bounds to prevent crashes. Font sizes and banner dimensions scale proportionally with frame width so the overlay looks correct at any resolution.

---

## Roadmap

- [ ] Semantic segmentation (SegFormer-b2) — road mask to restrict lane detection and depth decisions to drivable surface only
- [ ] Time-to-collision (TTC) — replace raw depth with seconds-until-impact based on bbox growth rate across frames
- [ ] Traffic light detection — red/green classifier integrated into decision engine
- [ ] Bird-eye view (BEV) perspective transform — more accurate lane fitting
- [ ] Kalman filter bbox smoothing — reduce jitter before speed estimation
- [ ] TensorRT export — 3–5× speed boost for YOLO and MiDaS
- [ ] Camera calibration — real-world speed in km/h instead of pixels/second
- [ ] Detection logger — save tracks, speeds, and decisions to CSV/JSON for offline analysis
- [ ] Lane departure warning — integrated into decision engine using BEV lane positions
- [ ] Multi-camera fusion — front and side camera combined perception

---


## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Intel MiDaS](https://github.com/isl-org/MiDaS)
- [deep_sort_realtime](https://github.com/levan92/deep_sort_realtime)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)

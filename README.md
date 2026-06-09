<<<<<<< HEAD
# Real-time Hybrid Autonomous Driving System

A real-time autonomous driving perception system built for the Digital Image Processing course (EC312) at NUST College of E&ME. The system fuses classical computer vision with deep learning to produce speed and steering decisions from live video — running entirely on a standard PC with no specialized hardware.

---

## What It Does

Two parallel pipelines run simultaneously on each video frame:

- **Lane Detection** — Canny edge detection + Hough Probabilistic Transform + EMA temporal smoothing to track left/right road boundaries and compute lane center offset
- **Object Detection** — YOLOv8s detects 9 road-relevant classes (persons, vehicles, animals), classifies each object as VERY NEAR / NEAR / FAR / VERY FAR by bounding box ratio, and checks whether it falls inside a trapezoidal road zone polygon

A decision module fuses both outputs into:
- **Speed command:** STOP / SLOW DOWN / MOVE FORWARD
- **Steering command:** STRAIGHT / TURN LEFT (mild/sharp) / TURN RIGHT (mild/sharp)

The system was later extended to a **multi-stream mode** processing three simultaneous camera feeds (FRONT, LEFT, RIGHT) with per-stream detectors, a shared YOLO background thread, cross-camera fused decisions, and a 2x2 composite grid display.

---

## Sample Output

### Lane Detection
![Lane Detection](assets/lane_detection.png)

### Object Detection with Road Zone
![Object Detection](assets/object_detection.png)

### Full HUD — Move Forward
![HUD Move Forward](assets/hud_move_forward.png)

### STOP Scenario — Person on Road
![STOP Scenario](assets/hud_stop.png)

### Multi-Stream 2x2 Grid
![Multi Stream](assets/multistream_grid.png)

---

## System Architecture

### Single-Stream Pipeline

```
Video Frame
    |
    +---> LaneDetection.py  (main thread)
    |         Canny -> Hough -> polyfit -> EMA -> lane_info
    |
    +---> object_detection.py  (YOLO daemon thread)
              YOLOv8s -> road zone check -> proximity tier -> detections
                  |
                  v
           decision_making.py
              speed_decision + steering_direction
                  |
                  v
           HUD overlay -> imshow
```

### Multi-Stream Extension

```
main.py
    |
    +-- [1] Single-Stream --> autonomous_car.py
    |
    +-- [2] Multi-Stream  --> stream_manager.py
                                |
                +---------------+---------------+
                |               |               |
          LaneDetector    SideDetector    SideDetector
            (FRONT)          (LEFT)         (RIGHT)
                |               |               |
                +-------+-------+               |
                        |                       |
                  Shared YOLO Thread  <----------+
                  (tagged queue, thread-safe dict)
                        |
                  fused_decide()
                        |
                  2x2 Grid Display
```

---

## Modules

| File | Role |
|------|------|
| `autonomous_car.py` | Single-stream entry point. Orchestrates video I/O, threading, HUD, logging |
| `LaneDetection.py` | Canny + Hough + EMA lane detection pipeline |
| `object_detection.py` | YOLOv8s inference, road zone polygon, proximity estimation |
| `decision_making.py` | Fuses lane offset + detections into speed/steering commands |
| `multi_stream/main.py` | Mode selection menu — launches single or multi-stream |
| `multi_stream/stream_manager.py` | Multi-stream orchestrator, YOLO thread, grid compositor, CSV logging |
| `multi_stream/lane_detector.py` | Class-based wrapper around lane detection (independent EMA per instance) |
| `multi_stream/side_detector.py` | Triangular-ROI boundary detector for lateral cameras |
| `multi_stream/config.py` | Shared constants, frame dimensions, stream-to-video mapping |

---

## Key Design Decisions

**Why decouple YOLO into a daemon thread?**
YOLOv8s inference takes ~30-150ms per frame depending on hardware. Running it in the main loop would cap display FPS to the inference rate. The background thread with a `maxsize=1` queue means the main loop always has the latest detection result without blocking, keeping the HUD smooth.

**Why EMA smoothing on lane lines?**
Raw Hough output fluctuates frame-to-frame due to lighting changes and partial occlusion. EMA (alpha=0.82) blends the current fit with the previous position, removing jitter while still tracking genuine road curvature. A miss counter (15 frames) resets the state if the lane disappears entirely.

**Why bounding box height ratio for distance?**
No depth sensor. The ratio of bounding box height to frame height is a reliable monocular proxy for distance on flat road surfaces. Thresholds 0.55 / 0.28 / 0.12 map to VERY NEAR / NEAR / FAR / VERY FAR and were tuned against real college road footage.

**Why a separate SideDetector for lateral cameras?**
Side cameras only see one road boundary at a time, and that boundary has a specific slope direction depending on camera side. Reusing the front LaneDetector would produce incorrect fits. SideDetector uses a triangular ROI and slope-aware filtering (LEFT: slope in (-2.5, -0.1), RIGHT: slope in (0.1, 2.5)) with top-5 line averaging for stability.

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| ROI_TOP_RATIO | 0.35 | Vertical start of trapezoidal ROI |
| HOUGH_THRESHOLD | 35 | Minimum votes per Hough line |
| MIN_LINE_LEN | 55 px | Minimum segment length |
| MAX_LINE_GAP | 80 px | Max gap between collinear segments |
| SMOOTH_ALPHA (EMA) | 0.82 | Lane smoothing weight |
| MISS_LIMIT | 15 frames | Frames before EMA state resets |
| Confidence (human/animal) | 0.40 | YOLO threshold |
| Confidence (vehicle) | 0.45 | YOLO threshold |
| Mild steering threshold | 30 px | Lane center offset |
| Sharp steering threshold | 80 px | Lane center offset |

---

## Performance

Tested on college road footage (daytime, shadows, curves, partial occlusion, multiple simultaneous objects).

| Condition | Observed FPS |
|-----------|-------------|
| Lane only (main thread) | 38-42 FPS |
| Lane + YOLO (decoupled) | 15-30 FPS |
| Multi-stream (3 cameras) | 6-8 FPS per stream |

FPS varies by scene complexity and number of detections. The decoupled threading architecture ensures the display loop never drops below real-time even when YOLO inference is slow.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/Hybrid-Autonomous-Driving-System
cd Hybrid-Autonomous-Driving-System
pip install -r requirements.txt
```

Place your `.mp4` video files in a `Videos/` folder.

**Single-stream:**
```bash
python autonomous_car.py
```

**Multi-stream (with mode menu):**
```bash
python multi_stream/main.py
```

**Keyboard controls:** `Q` quit | `P` pause | `N` next video

---

## Requirements

```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
```

Python 3.8+ on Windows/Linux. YOLOv8s weights (`yolov8s.pt`) download automatically on first run via Ultralytics.

---

## Detected Classes

Persons (priority: CRITICAL), cars/motorcycles/buses/trucks (HIGH), dogs/horses/sheep/cows (MEDIUM). Only objects whose bottom-center pixel falls within the road zone polygon trigger speed decisions.

---

## Challenges Solved

| Challenge | Solution |
|-----------|----------|
| Lane flickering | EMA smoothing (alpha=0.82) on line endpoints |
| YOLO latency blocking display | Daemon thread + maxsize=1 queue |
| Shadow-induced false lane lines | Slope filtering + ROI masking |
| Distance without depth sensor | Bounding box height ratio as proxy |
| On-road vs roadside classification | Bottom-center pixel tested against road zone polygon |
| Side camera single boundary | Dedicated SideDetector with triangular ROI and slope-aware filtering |

---

## Future Work

- Curved lane detection via sliding window polynomial fit
- Traffic sign / traffic light integration
- Monocular depth estimation (MiDaS) to replace the height-ratio proxy
- Adaptive ROI responding to road curvature
- Deployment on NVIDIA Jetson Nano for in-vehicle testing

---

## Team

| Name | Reg. ID |
|------|---------|
| Aleena Zia | 458086 |
| Aliza Haider | 482337 |
| Mahad Umar Qaisrani | 482972 |
| Kallem Ullah | 479688 |

Department of Computer and Software Engineering, College of E&ME, NUST
Course: EC312 Digital Image Processing | Supervisor: Dr. Asad

---

## References

- Ultralytics YOLOv8 — https://docs.ultralytics.com
- OpenCV Hough Line Transform — https://docs.opencv.org
- Canny, J. (1986). A Computational Approach to Edge Detection. IEEE Trans. PAMI
- Redmon et al. (2016). You Only Look Once. CVPR
=======
# Hybrid-Autonomous-Driving-System
>>>>>>> d159c002396e622ddd2c662ca8d3a8a54f3b5392

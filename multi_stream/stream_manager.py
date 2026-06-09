import cv2
import os
import sys
import time
import threading
import queue
import numpy as np

"""
Manages N video streams, runs YOLO in a shared background
thread, and composites all results into a 2x2 grid window.
"""

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from project import decision_making, object_detection

from lane_detector import LaneDetector
from side_detector import SideDetector

from config import (
    VIDEO_FOLDER, FRAME_WIDTH, FRAME_HEIGHT,
    GRID_CELL_W, GRID_CELL_H, STREAM_CONFIG, LOG_FILE,
)

class StreamState:
    """Holds everything for one camera stream."""

    def __init__(self, stream_id, name, video_files):
        self.stream_id = stream_id
        self.name = name
        self.video_files = video_files
        self.video_index = 0
        self.cap = None

        if name == "FRONT":
            self.detector = LaneDetector()
        else:
            self.detector = SideDetector(side=name)

        # runtime counters
        self.frame_count = 0
        self.max_frames = None
        self.fps = 0.0
        self.prev_time = time.time()
        self.finished = False

        # latest results
        self.current_frame = None
        self.lane_frame = None
        self.edge_map = None
        self.detections = []
        self.lane_info = {"left_line": None, "right_line": None,
                          "lane_center": FRAME_WIDTH // 2}
        self.side_info = {"boundary_line": None, "road_visible": False,
                          "boundary_offset": 0, "side": name}
        self.decision = "MOVE FORWARD"
        self.steering = "STRAIGHT"
        self.decision_color = (0, 255, 0)

    def open_next_video(self):
        """Open the next video in the stream"""
        if self.cap is not None:
            self.cap.release()

        if self.video_index >= len(self.video_files):
            self.finished = True
            return False

        path = self.video_files[self.video_index]
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            print(f"  [{self.name}] Cannot open: {path} — skipping")
            self.video_index += 1
            return self.open_next_video()

        self.detector.reset()
        self.frame_count = 0
        print(f"  [{self.name}] Now playing: {os.path.basename(path)}")
        self.video_index += 1
        return True

yolo_queue = queue.Queue(maxsize=3)
yolo_lock = threading.Lock()
yolo_results = {}

def yolo_worker():
    """
    Single background thread — processes tagged frames
    from all streams through the YOLO model sequentially.
    """
    while True:
        item = yolo_queue.get()
        if item is None:
            print("[YOLO Thread] Stopped.")
            break

        stream_id, frame = item
        detections, _ = object_detection.process_frame(frame, True)
        with yolo_lock:
            yolo_results[stream_id] = detections


def text_bg(img, text, pos, fg, bg):
    x, y = pos
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (x - 2, y - h - 2), (x + w + 2, y + 2), bg, -1)
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, fg, 1)

def draw_boxes(frame, detections):
    """Draw clean YOLO bounding boxes with simple labels."""
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        if not d["on_road"]:
            clr, th = (180, 180, 180), 1
        elif d["distance"] == "VERY NEAR":
            clr, th = (0, 0, 255), 2
        elif d["distance"] == "NEAR":
            clr, th = (0, 140, 255), 2
        elif d["distance"] == "FAR":
            clr, th = (0, 200, 100), 1
        else:
            clr, th = (200, 200, 0), 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), clr, th)

        if d["on_road"]:
            label = f"{d['label']} | {d['distance']}"
        else:
            label = d["label"]
        text_bg(frame, label, (x1, max(y1 - 4, 12)), clr, (0, 0, 0))
    return frame


def annotate(stream):
    """Build annotated cell for one stream."""
    frame = stream.current_frame.copy()

    if stream.lane_frame is not None:
        mask = cv2.cvtColor(stream.lane_frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
        colored = cv2.bitwise_and(stream.lane_frame, stream.lane_frame,
                                  mask=mask)
        frame = cv2.addWeighted(frame, 1.0, colored, 0.6, 0)

    frame = draw_boxes(frame, stream.detections)

    text_bg(frame, stream.name, (5, 15), (0, 255, 255), (0, 0, 0))
    text_bg(frame, stream.decision, (5, 33), stream.decision_color, (0, 0, 0))
    text_bg(frame, stream.steering, (5, 51), (255, 255, 255), (0, 0, 0))

    if stream.name == "FRONT":
        threat = object_detection.get_nearest_threat(stream.detections)
        if threat:
            banner = f"!! {threat['label'].upper()} — {threat['distance']} !!"
            text_bg(frame, banner,
                    (frame.shape[1] // 2 - 60, 15), (0, 0, 255), (0, 0, 0))

    return frame

def make_stats_panel(streams, cw, ch, fused_result=None):
    """Bottom-right cell: shows the FINAL fused decision + per-stream status."""
    panel = np.zeros((ch, cw, 3), dtype=np.uint8)

    cv2.rectangle(panel, (0, 0), (cw - 1, ch - 1), (40, 40, 40), -1)
    cv2.rectangle(panel, (0, 0), (cw - 1, ch - 1), (80, 80, 80), 1)
    y = 20

    if fused_result:
        final_decision, final_steering, final_color = fused_result

        text_bg(panel, "FINAL COMMAND", (10, y), (0, 255, 255), (30, 30, 30))
        y += 22

        # draw the decision in a bigger font
        cv2.putText(panel, final_decision, (10, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, final_color, 2)

        y += 25
        text_bg(panel, f"Steering: {final_steering}", (10, y),
                (255, 255, 255), (30, 30, 30))
        y += 8

    # separator line
    y += 12
    cv2.line(panel, (10, y), (cw - 10, y), (80, 80, 80), 1)
    y += 15

    # per-stream status
    text_bg(panel, "PER-CAMERA STATUS", (10, y), (180, 180, 180), (30, 30, 30))
    for s in streams:
        y += 18
        on_road = sum(1 for d in s.detections if d["on_road"])
        status = "DONE" if s.finished else s.decision
        color = (120, 120, 120) if s.finished else s.decision_color
        text_bg(panel, f"{s.name}: {status}  ({on_road} on road)",
                (10, y), color, (30, 30, 30))

    # controls
    y += 25
    text_bg(panel, "Q=Quit  P=Pause  N=Skip",
            (10, y), (120, 120, 120), (30, 30, 30))

    return panel


def build_grid(streams, fused_result=None):
    """
    Build a 2x2 grid:
        FRONT  |  STATS
        LEFT   |  RIGHT
    """
    cw, ch = GRID_CELL_W, GRID_CELL_H
    grid = np.zeros((ch * 2, cw * 2, 3), dtype=np.uint8)

    layout = {"FRONT": (0, 0), "LEFT": (0, ch), "RIGHT": (cw, ch)}

    for stream in streams[:3]:
        if stream.name not in layout:
            continue
        if stream.current_frame is not None:
            cell = annotate(stream)
            cell = cv2.resize(cell, (cw, ch))
        else:
            cell = np.zeros((ch, cw, 3), dtype=np.uint8)
            cv2.putText(cell, f"{stream.name} -- waiting",
                        (10, ch // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)

        x, y = layout[stream.name]
        grid[y:y + ch, x:x + cw] = cell

    grid[0:ch, cw:cw * 2] = make_stats_panel(streams, cw, ch, fused_result)
    return grid


def get_sync_limits(streams):
    """
    For each video-set index, find the MINIMUM frame count
    across all streams so they stay synchronized.
    """
    max_vids = max(len(s.video_files) for s in streams)
    limits = []
    for vi in range(max_vids):
        counts = []
        for s in streams:
            if vi < len(s.video_files):
                cap = cv2.VideoCapture(s.video_files[vi])
                counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                cap.release()
        limits.append(min(counts) if counts else 0)
    return limits

def run():
    """Entry point for multi-stream mode."""

    streams = []
    for idx, (name, cfg) in enumerate(STREAM_CONFIG.items()):
        videos = [os.path.join(VIDEO_FOLDER, v) for v in cfg["videos"]]
        streams.append(StreamState(idx, name, videos))

    sync_limits = get_sync_limits(streams)
    print(f"[Sync] Frame limits per video set: {sync_limits}")

    for s in streams:
        s.open_next_video()
        if s.video_index - 1 < len(sync_limits):
            s.max_frames = sync_limits[s.video_index - 1]

    t = threading.Thread(target=yolo_worker, daemon=True)
    t.start()
    print("[Multi-Stream] YOLO worker started.")

    log = open(LOG_FILE, "w")
    log.write("stream,frame,fps,decision,steering,on_road_count\n")

    print("\nControls: Q = Quit | P = Pause | N = Skip current videos")
    print("─" * 50)

    while True:
        # check if every stream is done
        if all(s.finished for s in streams):
            print("\nAll streams finished.")
            break

        # process one frame per stream
        all_at_limit = True
        for s in streams:
            if s.finished or s.cap is None:
                continue

            if s.max_frames and s.frame_count >= s.max_frames:
                continue

            all_at_limit = False
            ret, frame = s.cap.read()

            if not ret:
                if not s.open_next_video():
                    continue
                vi = s.video_index - 1
                s.max_frames = sync_limits[vi] if vi < len(sync_limits) else None
                ret, frame = s.cap.read()
                if not ret:
                    s.finished = True
                    continue

            s.frame_count += 1
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            s.current_frame = frame.copy()

            if s.name == "FRONT":
                s.edge_map, s.lane_frame, s.lane_info = s.detector.process(frame)
            else:
                s.edge_map, s.lane_frame, s.side_info = s.detector.process(frame)

            if not yolo_queue.full():
                yolo_queue.put((s.stream_id, frame.copy()))

            with yolo_lock:
                s.detections = list(yolo_results.get(s.stream_id, []))

            s.decision, s.steering, s.decision_color = \
                decision_making.decide(s.detections, s.lane_info,
                                       FRAME_WIDTH, s.name)

            now = time.time()
            new_fps = 1.0 / max(now - s.prev_time, 0.001)
            s.fps = 0.9 * s.fps + 0.1 * new_fps
            s.prev_time = now

            if s.frame_count % 30 == 0:
                on_road = sum(1 for d in s.detections if d["on_road"])
                log.write(f"{s.name},{s.frame_count},{s.fps:.1f},"
                          f"{s.decision},{s.steering},{on_road}\n")

        if all_at_limit:
            for s in streams:
                if s.finished:
                    continue
                if not s.open_next_video():
                    continue
                vi = s.video_index - 1
                s.max_frames = sync_limits[vi] if vi < len(sync_limits) else None

        # find each stream by name
        front_s = next((s for s in streams if s.name == "FRONT"), None)
        left_s = next((s for s in streams if s.name == "LEFT"), None)
        right_s = next((s for s in streams if s.name == "RIGHT"), None)

        fused_result = None
        if front_s and left_s and right_s:
            fused_result = decision_making.fused_decide(
                front_s.detections,
                left_s.detections,
                right_s.detections,
                front_s.lane_info,
                FRAME_WIDTH,
            )

        grid = build_grid(streams, fused_result)
        cv2.imshow("Multi-Stream Autonomous Car", grid)

        key = cv2.waitKey(1)
        if key != -1:
            ch = chr(key & 0xFF).lower()
            if ch == "q":
                print("\nQuit by user.")
                break
            elif ch == "p":
                print("PAUSED — press P to resume")
                while True:
                    k = cv2.waitKey(0)
                    if chr(k & 0xFF).lower() == "p":
                        print("RESUMED")
                        break
            elif ch == "n":
                for s in streams:
                    if not s.finished:
                        s.open_next_video()
                        vi = s.video_index - 1
                        s.max_frames = sync_limits[vi] if vi < len(sync_limits) else None

    yolo_queue.put(None)
    for s in streams:
        if s.cap:
            s.cap.release()

    log.close()
    cv2.destroyAllWindows()
    print("Multi-stream session complete.")

import os

# Paths 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
VIDEO_FOLDER = os.path.join(PROJECT_ROOT, "multi_stream/Videos")
YOLO_MODEL = os.path.join(PROJECT_ROOT, "yolov8s.pt")
LOG_FILE = os.path.join(os.path.dirname(__file__), "multi_stream_log.txt")

# Frame settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# Grid display
GRID_CELL_W = 540
GRID_CELL_H = 300

# Assign video files to each camera stream
STREAM_CONFIG = {
    "FRONT": {
        "videos": ["ms_video01s.mp4", "ms_video02s.mp4"],
    },
    "LEFT": {
        "videos": ["ms_video01l.mp4", "ms_video02l.mp4"],
    },
    "RIGHT": {
        "videos": ["ms_video01r.mp4", "ms_video02r.mp4"],
    },
}

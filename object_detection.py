import cv2
import numpy as np
import time
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

# classes we care about: id -> (name, type, color, priority)
ROAD_HAZARD_CLASSES = {
    0:  ('person',       'human',   (0, 200, 255),  'CRITICAL'),
    1:  ('bicycle',      'vehicle', (0, 255, 100),  'HIGH'),
    2:  ('car',          'vehicle', (0, 255, 100),  'HIGH'),
    3:  ('motorcycle',   'vehicle', (0, 255, 100),  'HIGH'),
    5:  ('bus',          'vehicle', (0, 180, 80),   'HIGH'),
    6:  ('train',        'vehicle', (0, 140, 60),   'HIGH'),
    7:  ('truck',        'vehicle', (0, 180, 80),   'HIGH'),
    14: ('bird',         'animal',  (255, 165, 0),  'LOW'),
    15: ('cat',          'animal',  (255, 140, 0),  'MEDIUM'),
    16: ('dog',          'animal',  (255, 120, 0),  'MEDIUM'),
    17: ('horse',        'animal',  (255, 100, 0),  'HIGH'),
    18: ('sheep',        'animal',  (255, 100, 0),  'HIGH'),
    19: ('cow',          'animal',  (255, 80,  0),  'HIGH'),
    9:  ('traffic light','sign',    (255, 255, 0),  'MEDIUM'),
    11: ('stop sign',    'sign',    (0,   0, 255),  'HIGH'),
    12: ('parking meter','sign',    (200, 200, 0),  'LOW'),
    24: ('backpack',     'object',  (200, 100, 255),'LOW'),
    25: ('umbrella',     'object',  (200, 100, 255),'LOW'),
    28: ('suitcase',     'object',  (200, 100, 255),'LOW'),
    58: ('potted plant', 'object',  (0,  200, 100), 'MEDIUM'),
}

CONFIDENCE_THRESHOLDS = {
    'human':   0.40,
    'vehicle': 0.45,
    'animal':  0.40,
    'sign':    0.45,
    'object':  0.50,
}

PRIORITY_ORDER = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}

# process every 3rd frame to keep fps smooth
FRAME_SKIP = 3


def get_road_zone(frame, camera_angle='normal'):
    h, w = frame.shape[:2]

    if camera_angle == 'downward':
        # camera tilted down so road starts higher in the frame
        zone = np.array([
            [int(w * 0.20), h],
            [int(w * 0.80), h],
            [int(w * 0.65), int(h * 0.15)],
            [int(w * 0.35), int(h * 0.15)],
        ], dtype=np.int32)
    else:
        # normal forward camera
        zone = np.array([
            [int(w * 0.28), h],
            [int(w * 0.72), h],
            [int(w * 0.58), int(h * 0.40)],
            [int(w * 0.42), int(h * 0.40)],
        ], dtype=np.int32)

    return zone


def is_in_road_zone(box, road_zone, frame_shape):
    x1, y1, x2, y2 = box
    # check bottom center because thats where feet/wheels touch the ground
    bx = (x1 + x2) // 2
    by = y2
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [road_zone], 255)
    if 0 <= by < frame_shape[0] and 0 <= bx < frame_shape[1]:
        return mask[by, bx] == 255
    return False


def estimate_distance(box, frame_shape):
    x1, y1, x2, y2 = box
    fh, fw = frame_shape[:2]
    bh = y2 - y1
    cx = (x1 + x2) // 2
    ratio = bh / fh

    # bigger box = closer object
    if ratio > 0.55:
        lvl, col = 'VERY NEAR', (0, 0, 255)
    elif ratio > 0.28:
        lvl, col = 'NEAR', (0, 100, 255)
    elif ratio > 0.12:
        lvl, col = 'FAR', (0, 200, 100)
    else:
        lvl, col = 'VERY FAR', (160, 160, 160)

    off = cx - fw // 2
    if abs(off) < fw * 0.12:
        pos = 'CENTER'
    elif off < 0:
        pos = 'LEFT'
    else:
        pos = 'RIGHT'

    return lvl, col, pos


def detect_cones(frame, road_zone):
    h, w = frame.shape[:2]
    cones = []

    # only look inside the road zone
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [road_zone], 255)
    road_only = cv2.bitwise_and(frame, frame, mask=mask)

    hsv = cv2.cvtColor(road_only, cv2.COLOR_BGR2HSV)

    # orange range
    lo_org = np.array([5,  100, 100])
    hi_org = np.array([25, 255, 255])

    # red wraps around in hsv so need two ranges
    lo_r1 = np.array([0,   100, 100])
    hi_r1 = np.array([10,  255, 255])
    lo_r2 = np.array([160, 100, 100])
    hi_r2 = np.array([180, 255, 255])

    m_org = cv2.inRange(hsv, lo_org, hi_org)
    m_r1  = cv2.inRange(hsv, lo_r1,  hi_r1)
    m_r2  = cv2.inRange(hsv, lo_r2,  hi_r2)
    cmask = m_org | m_r1 | m_r2

    # clean up noise
    k     = np.ones((5, 5), np.uint8)
    cmask = cv2.morphologyEx(cmask, cv2.MORPH_OPEN,  k)
    cmask = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500 or area > 25000:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)

        # cones are taller than wide
        if ch < cw:
            continue

        # ignore anything too high up in frame
        if y < h * 0.25:
            continue

        cones.append((x, y, x + cw, y + ch))

    return cones


def detect_unknown_obstacles(frame, road_zone, known_boxes):
    h, w = frame.shape[:2]
    unknowns = []

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [road_zone], 255)
    road_only = cv2.bitwise_and(frame, frame, mask=mask)

    gray    = cv2.cvtColor(road_only, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges   = cv2.Canny(blurred, 60, 150)

    k     = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, k, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4000:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)

        if cw > w * 0.50:   # too wide = probably road surface
            continue
        if y < h * 0.35:    # too high = background
            continue
        if ch < 25:
            continue
        if cw > ch * 3:     # too flat to be a real obstacle
            continue

        ux1, uy1, ux2, uy2 = x, y, x + cw, y + ch

        # skip if already detected by yolo
        overlap = False
        for (kx1, ky1, kx2, ky2) in known_boxes:
            if not (ux2 < kx1 or ux1 > kx2 or uy2 < ky1 or uy1 > ky2):
                overlap = True
                break

        if not overlap:
            unknowns.append((ux1, uy1, ux2, uy2))

    return unknowns


def draw_text(img, text, pos, scale=0.55, color=(255, 255, 255), thickness=2):
    x, y = pos
    # black outline so text is readable on any background
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def process_frame(frame, run_layer2=True, camera_angle='normal'):
    detections  = []
    known_boxes = []
    annotated   = frame.copy()

    road_zone = get_road_zone(frame, camera_angle)

    # draw road zone with slight tint
    overlay = annotated.copy()
    cv2.fillPoly(overlay, [road_zone], (255, 255, 100))
    cv2.addWeighted(overlay, 0.07, annotated, 0.93, 0, annotated)
    cv2.polylines(annotated, [road_zone], True, (0, 255, 255), 2)

    # yolo detection
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())

            if cls_id not in ROAD_HAZARD_CLASSES:
                continue

            label, cat, base_col, priority = ROAD_HAZARD_CLASSES[cls_id]
            conf      = box.conf[0].item()
            threshold = CONFIDENCE_THRESHOLDS.get(cat, 0.45)

            if conf < threshold:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            in_road  = is_in_road_zone((x1, y1, x2, y2), road_zone, frame.shape)
            dist, dcol, pos = estimate_distance((x1, y1, x2, y2), frame.shape)

            if in_road:
                draw_col  = dcol
                zone_lbl  = 'ON ROAD'
                thickness = 3
            else:
                draw_col  = (0, 200, 255)
                zone_lbl  = 'ROADSIDE'
                thickness = 1

            known_boxes.append((x1, y1, x2, y2))
            detections.append({
                'label':      label,
                'category':   cat,
                'confidence': round(conf, 2),
                'box':        (x1, y1, x2, y2),
                'distance':   dist,
                'position':   pos,
                'on_road':    in_road,
                'priority':   priority,
            })

            cv2.rectangle(annotated, (x1, y1), (x2, y2), draw_col, thickness)
            draw_text(annotated,
                      f"{label} {conf:.2f} | {dist} | {zone_lbl}",
                      (x1, max(y1 - 12, 16)),
                      color=draw_col)

            if in_road:
                draw_text(annotated, f"[{pos}]",
                          (x1, max(y1 + 8, 30)),
                          scale=0.45, color=draw_col)

    # cone detection runs with layer 2
    if run_layer2:
        cones = detect_cones(frame, road_zone)
        for (cx1, cy1, cx2, cy2) in cones:
            cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 100, 255), 2)
            draw_text(annotated, "CONE",
                      (cx1, max(cy1 - 8, 14)),
                      scale=0.5, color=(0, 100, 255))
            detections.append({
                'label':      'cone',
                'category':   'object',
                'confidence': 0.70,
                'box':        (cx1, cy1, cx2, cy2),
                'distance':   estimate_distance((cx1, cy1, cx2, cy2), frame.shape)[0],
                'position':   estimate_distance((cx1, cy1, cx2, cy2), frame.shape)[2],
                'on_road':    True,
                'priority':   'HIGH',
            })

        unknowns = detect_unknown_obstacles(frame, road_zone, known_boxes)
        for (ux1, uy1, ux2, uy2) in unknowns[:2]:
            cv2.rectangle(annotated, (ux1, uy1), (ux2, uy2), (0, 0, 200), 2)
            draw_text(annotated, "UNKNOWN OBSTACLE",
                      (ux1, max(uy1 - 8, 14)),
                      scale=0.45, color=(0, 0, 200))

    # sort so on-road and highest priority come first
    detections.sort(
        key=lambda d: (d['on_road'], PRIORITY_ORDER.get(d['priority'], 0)),
        reverse=True
    )

    return detections, annotated

def get_nearest_threat(detections):
    on_road = [d for d in detections if d['on_road']]
    if not on_road:
        return None
    return sorted(on_road, key=lambda x: PRIORITY_ORDER[x['priority']], reverse=True)[0]
import cv2
import numpy as np

# =====================================================
# Lane Detection
# =====================================================

ROI_TOP_RATIO = 0.35
HOUGH_THRESHOLD = 35
MIN_LINE_LEN = 55
MAX_LINE_GAP = 80
MISS_LIMIT = 15
SMOOTH_ALPHA = 0.82
Y_TOP_FLOOR = 0.38
prev_left  = None
prev_right = None
left_miss  = 0
right_miss = 0

# =====================================================
# EMA SMOOTHING
# =====================================================

def reset_state():
    """Resets EMA and miss counters between videos."""
    global prev_left, prev_right, left_miss, right_miss
    prev_left  = None
    prev_right = None
    left_miss  = 0
    right_miss = 0

def ema(old, new, alpha=SMOOTH_ALPHA):
    """
        Exponential Moving Average smoothing for lane coordinates.
        Reduces jitter between frames by blending old and new detections.
        Higher alpha = more weight to previous frame.
    """
    if old is None:
        return new
    return tuple(int(alpha * o + (1 - alpha) * n) for o, n in zip(old, new))


# =====================================================
# ROI MASK
# =====================================================

def region_mask(img):
    """
        Applies a trapezoidal mask to isolate the region of interest (ROI).
        Removes sky, trees, and irrelevant surroundings from the image.
        Works for both grayscale and BGR images.
    """
    h, w = img.shape[:2]

    pts = np.array([[
        (int(w * 0.02), h),
        (int(w * 0.98), h),
        (int(w * 0.75), int(h * 0.35)),  # wider and higher
        (int(w * 0.25), int(h * 0.35))
    ]], dtype=np.int32)

    mask = np.zeros_like(img)

    if len(img.shape) == 2:
        cv2.fillPoly(mask, pts, 255)
    else:
        cv2.fillPoly(mask, pts, (255, 255, 255))

    return cv2.bitwise_and(img, mask)


# =====================================================
# EDGE DETECTION
# =====================================================

def get_edges(frame):
    """
        Converts frame to grayscale, applies Gaussian blur, then Canny edge detection.
        Returns two edge maps: full-frame (for display) and ROI-masked (for Hough detection).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges_full = cv2.Canny(blur, 60, 120)
    edges_roi  = region_mask(edges_full)
    return edges_full, edges_roi

# =====================================================
# FIT LINE
# =====================================================

def fit_line(lines, h, w, roi_top):
    """
        Fits a single best-fit line through a set of Hough line segments using polyfit.
        Returns (x_bottom, y_bottom, x_top, y_top) representing the lane boundary line.
        y_top is clamped to avoid lines extending too high above the road surface.
    """
    if len(lines) == 0:
        return None

    xs = []
    ys = []
    weights = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        xs.extend([x1, x2])
        ys.extend([y1, y2])
        w1 = (y1 / h) ** 2
        w2 = (y2 / h) ** 2
        weights.extend([w1, w2])

    coefficient = np.polyfit(ys, xs, 1, w=weights) # weighted fit

    y_bottom = h
    ys_arr = np.array(ys)
    upper_ys = ys_arr[ys_arr < np.median(ys_arr)]

    if len(upper_ys) > 0:
        y_top = int(max(np.median(upper_ys), roi_top, h * Y_TOP_FLOOR))
    else:
        y_top = int(h * 0.7)

    x_bottom = int(np.clip(np.polyval(coefficient, y_bottom), 0, w - 1))
    x_top    = int(np.clip(np.polyval(coefficient, y_top), 0, w - 1))

    return x_bottom, y_bottom, x_top, y_top


# =====================================================
# MAIN DETECTION
# =====================================================

def detect_boundary(frame):
    """
        Main detection function. Runs edge detection and Hough transform, then
        separates lines into left/right based on slope and position relative to midpoint.
        Returns fitted left/right lane tuples and edge map.
    """
    edges_full, edges_roi = get_edges(frame)

    lines = cv2.HoughLinesP(
        edges_roi,
        1,
        np.pi / 180,
        HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LEN,
        maxLineGap=MAX_LINE_GAP
    )

    h, w = frame.shape[:2]
    mid = w // 2
    roi_top = int(h * ROI_TOP_RATIO)

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x1 == x2:
                continue

            slope = (y2 - y1) / (x2 - x1)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # left side
            if -1.5 < slope < -0.2 and (x1 + x2) // 2 < mid:
                weight = length + (mid - (x1 + x2) // 2) * 0.5
                left_lines.append((line, weight))

            # right side
            elif 0.2 < slope < 1.5 and (x1 + x2) // 2 > mid:
                weight = length + (((x1 + x2) // 2) - mid) * 0.5
                right_lines.append((line, weight))

    # prioritize longest lines
    left_lines  = sorted(left_lines,  key=lambda x: x[1], reverse=True)[:8]
    right_lines = sorted(right_lines, key=lambda x: x[1], reverse=True)[:8]

    left_lines  = [x[0] for x in left_lines]
    right_lines = [x[0] for x in right_lines]

    left_lane  = fit_line(left_lines, h, w, roi_top)
    right_lane = fit_line(right_lines, h, w, roi_top)

    return left_lane, right_lane, edges_full


# =====================================================
# DRAW RESULTS
# =====================================================

def draw(frame, edges, left, right):
    """
        Draws detected lane lines on the frame.
        Applies EMA smoothing using previous frame detections to reduce flickering.
        Resets lanes after MISS_LIMIT consecutive missed detections.
        Returns edge map (with overlaid lines).
    """
    global prev_left, prev_right, left_miss, right_miss

    if left is not None:
        prev_left = ema(prev_left, left)
        left_miss = 0
    else:
        left_miss += 1
        if left_miss > MISS_LIMIT:
            prev_left = None

    if right is not None:
        prev_right = ema(prev_right, right)
        right_miss = 0
    else:
        right_miss += 1
        if right_miss > MISS_LIMIT:
            prev_right = None

    left = prev_left
    right = prev_right

    result = frame.copy()

    # ---------- edge map ----------
    edge_map = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if left:
        cv2.line(edge_map,
                 (left[0], left[1]),
                 (left[2], left[3]),
                 (0, 0, 255), 3)

    if right:
        cv2.line(edge_map,
                 (right[0], right[1]),
                 (right[2], right[3]),
                 (0, 0, 255), 3)

    # ---------- final output ----------
    if left:
        cv2.line(result,
                 (left[0], left[1]),
                 (left[2], left[3]),
                 (0, 0, 255), 4)

    if right:
        cv2.line(result,
                 (right[0], right[1]),
                 (right[2], right[3]),
                 (0, 0, 255), 4)

    return edge_map, result


# =====================================================
# PROCESS FRAME
# =====================================================

def process(frame):
    """
        Top-level function called per frame. Runs detection and drawing pipeline,
        then computes lane_center x-coordinate at the bottom of the frame.
        Returns edge_map, annotated result, and lane_info dict for downstream use.
    """
    h, w = frame.shape[:2]
    left, right, edges = detect_boundary(frame)
    edge_map, result = draw(frame, edges, left, right)

    l = prev_left
    r = prev_right

    # lane center x at bottom of frame
    if isinstance(l, tuple) and isinstance(r, tuple):
        lane_center = (int(l[0]) + int(r[0])) // 2
    elif isinstance(l, tuple):
        lane_center = int(l[0] + (w * 0.25))
    elif isinstance(r, tuple):
        lane_center = int(r[0] - (w * 0.25))
    else:
        lane_center = w // 2

    lane_info = {
        "left_line": prev_left,  # (x_bottom, y_bottom, x_top, y_top)
        "right_line": prev_right,
        "lane_center": lane_center
    }

    return edge_map, result, lane_info


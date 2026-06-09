import cv2
import numpy as np

class LaneDetector:
    """Per-stream lane detector with independent EMA state."""

    ROI_TOP_RATIO = 0.35
    HOUGH_THRESHOLD = 35
    MIN_LINE_LEN = 55
    MAX_LINE_GAP = 80
    MISS_LIMIT = 15
    SMOOTH_ALPHA = 0.82
    Y_TOP_FLOOR = 0.38

    def __init__(self):
        self.prev_left = None
        self.prev_right = None
        self.left_miss = 0
        self.right_miss = 0

    def reset(self):
        """Call between videos to clear EMA memory."""
        self.prev_left = None
        self.prev_right = None
        self.left_miss = 0
        self.right_miss = 0

    def process(self, frame):
        """
        Top-level per-frame function.
        Returns: (edge_map_bgr, annotated_frame, lane_info_dict)
        """
        h, w = frame.shape[:2]
        left, right, edges = self.detect_boundary(frame)
        edge_map, result = self.draw(frame, edges, left, right)

        # Compute lane center at bottom of frame
        l, r = self.prev_left, self.prev_right
        if isinstance(l, tuple) and isinstance(r, tuple):
            lane_center = (int(l[0]) + int(r[0])) // 2
        elif isinstance(l, tuple):
            lane_center = int(l[0] + (w * 0.25))
        elif isinstance(r, tuple):
            lane_center = int(r[0] - (w * 0.25))
        else:
            lane_center = w // 2

        lane_info = {
            "left_line": self.prev_left,
            "right_line": self.prev_right,
            "lane_center": lane_center,
        }
        return edge_map, result, lane_info

    def ema(self, old, new):
        """
        Exponential Moving Average smoothing for lane coordinates.
        Higher alpha = more weight to previous frame.
        """
        if old is None:
            return new
        a = self.SMOOTH_ALPHA

        return tuple(int(a * o + (1 - a) * n) for o, n in zip(old, new))

    @staticmethod
    def region_mask(img):
        """
        Applies a trapezoidal mask to isolate the region of interest (ROI).
        """
        h, w = img.shape[:2]
        pts = np.array([[
            (int(w * 0.02), h),
            (int(w * 0.98), h),
            (int(w * 0.75), int(h * 0.35)),
            (int(w * 0.25), int(h * 0.35)),
        ]], dtype=np.int32)
        mask = np.zeros_like(img)

        if len(img.shape) == 2:
            cv2.fillPoly(mask, pts, 255)
        else:
            cv2.fillPoly(mask, pts, (255, 255, 255))

        return cv2.bitwise_and(img, mask)

    def get_edges(self, frame):
        """
        Converts frame to grayscale, applies Gaussian blur, then Canny edge detection.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges_full = cv2.Canny(blur, 60, 120)
        edges_roi = self.region_mask(edges_full)

        return edges_full, edges_roi

    def fit_line(self, lines, h, w, roi_top):
        """
        Fits a single best-fit line through a set of Hough line segments using polyfit.
        """
        if len(lines) == 0:
            return None
        xs, ys, weights = [], [], []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            xs.extend([x1, x2])
            ys.extend([y1, y2])
            weights.extend([(y1 / h) ** 2, (y2 / h) ** 2])

        coefficient = np.polyfit(ys, xs, 1, w=weights)
        y_bottom = h
        ys_arr = np.array(ys)
        upper = ys_arr[ys_arr < np.median(ys_arr)]

        if len(upper) > 0:
            y_top = int(max(np.median(upper), roi_top, h * self.Y_TOP_FLOOR))
        else:
            y_top = int(h * 0.7)

        x_bottom = int(np.clip(np.polyval(coefficient, y_bottom), 0, w - 1))
        x_top    = int(np.clip(np.polyval(coefficient, y_top),    0, w - 1))

        return x_bottom, y_bottom, x_top, y_top

    def detect_boundary(self, frame):
        """
        Main detection function. Runs edge detection and Hough transform, then
        separates lines into left/right based on slope and position relative to midpoint.
        """
        edges_full, edges_roi = self.get_edges(frame)

        lines = cv2.HoughLinesP(
            edges_roi, 1, np.pi / 180,
            self.HOUGH_THRESHOLD,
            minLineLength=self.MIN_LINE_LEN,
            maxLineGap=self.MAX_LINE_GAP,
        )

        h, w = frame.shape[:2]
        mid = w // 2
        roi_top = int(h * self.ROI_TOP_RATIO)

        left_lines, right_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                cx = (x1 + x2) // 2

                if -1.5 < slope < -0.2 and cx < mid:
                    left_lines.append((line, length + (mid - cx) * 0.5))
                elif 0.2 < slope < 1.5 and cx > mid:
                    right_lines.append((line, length + (cx - mid) * 0.5))

        left_lines = [x[0] for x in sorted(left_lines,  key=lambda x: x[1], reverse=True)[:8]]
        right_lines = [x[0] for x in sorted(right_lines, key=lambda x: x[1], reverse=True)[:8]]

        return (
            self.fit_line(left_lines, h, w, roi_top),
            self.fit_line(right_lines, h, w, roi_top),
            edges_full,
        )

    def draw(self, frame, edges, left, right):
        """
        Draws detected lane lines on the frame.
        Applies EMA smoothing using previous frame detections.
        Resets lanes after MISS_LIMIT consecutive missed detections.
        """

        # Update EMA for left lane
        if left is not None:
            self.prev_left = self.ema(self.prev_left, left)
            self.left_miss = 0
        else:
            self.left_miss += 1
            if self.left_miss > self.MISS_LIMIT:
                self.prev_left = None

        # Update EMA for right lane
        if right is not None:
            self.prev_right = self.ema(self.prev_right, right)
            self.right_miss = 0
        else:
            self.right_miss += 1
            if self.right_miss > self.MISS_LIMIT:
                self.prev_right = None

        left = self.prev_left
        right = self.prev_right

        result = frame.copy()
        edge_map = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        for pt_set, img in [(left, edge_map), (left, result),
                            (right, edge_map), (right, result)]:
            if pt_set:
                thick = 3 if img is edge_map else 4
                cv2.line(img,
                         (pt_set[0], pt_set[1]),
                         (pt_set[2], pt_set[3]),
                         (0, 0, 255), thick)

        return edge_map, result

import cv2
import numpy as np

class SideDetector:
    """
    Detects the road boundary line for a LEFT or RIGHT camera.
    """

    HOUGH_THRESHOLD = 30
    MIN_LINE_LEN = 40
    MAX_LINE_GAP = 60
    SMOOTH_ALPHA = 0.8
    MISS_LIMIT = 15

    def __init__(self, side="LEFT"):
        """
        This tells which direction the road is on: "LEFT" or "RIGHT".
        """
        self.side = side
        self.prev_line = None
        self.miss_count = 0

    def reset(self):
        """
        Called when switching to a new video.
        """
        self.prev_line = None
        self.miss_count = 0

    def process(self, frame):
        """
        Detect the road boundary from a side camera frame.
        """
        h, w = frame.shape[:2]

        # Get edges using Canny
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 50, 120)

        # Mask edges to only look at the triangular road region
        masked_edges = self.apply_triangle_mask(edges, h, w)

        # Find lines using Hough Transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.HOUGH_THRESHOLD,
            minLineLength=self.MIN_LINE_LEN,
            maxLineGap=self.MAX_LINE_GAP,
        )

        # Pick the best boundary line
        best_line = self.find_best_line(lines)

        # Smooth with EMA
        if best_line is not None:
            self.prev_line = self.smooth_line(self.prev_line, best_line)
            self.miss_count = 0
        else:
            self.miss_count += 1
            if self.miss_count > self.MISS_LIMIT:
                self.prev_line = None

        # Draw the result
        edge_map = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        overlay = frame.copy()

        if self.prev_line is not None:
            x1, y1, x2, y2 = self.prev_line
            cv2.line(edge_map, (x1, y1), (x2, y2), (0, 0, 255), 5)
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 5)

        # Draw the triangle ROI outline
        triangle_pts = self.get_triangle_points(h, w)
        cv2.polylines(overlay, [triangle_pts], True, (255, 255, 0), 2)

        # Build the info dict
        side_info = self.build_info(w)

        return edge_map, overlay, side_info

    def get_triangle_points(self, h, w):
        """
        Return 3 corners of a TRIANGLE ROI on the road surface.
        """
        if self.side == "LEFT":
            pts = np.array([
                [int(w * 0.05), h],
                [int(w * 0.95), h],
                [int(w * 0.85), int(h * 0.40)],
            ], dtype=np.int32)
        else:
            pts = np.array([
                [int(w * 0.05), h],
                [int(w * 0.95), h],
                [int(w * 0.15), int(h * 0.40)],
            ], dtype=np.int32)

        return pts

    def apply_triangle_mask(self, edges, h, w):
        """
        Apply the triangle mask to the edge image.
        """
        mask = np.zeros_like(edges)
        pts = self.get_triangle_points(h, w)
        cv2.fillPoly(mask, [pts], 255)
        return cv2.bitwise_and(edges, mask)

    def find_best_line(self, lines):
        """
        From all the Hough lines, pick the one that looks most like
        a road boundary — a long diagonal line going in the right direction.
        """
        if lines is None:
            return None

        candidates = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x1 == x2:
                continue

            slope = (y2 - y1) / (x2 - x1)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if self.side == "LEFT":
                if -2.5 < slope < -0.1:
                    score = length
                    candidates.append((line[0], score))
            else:
                if 0.1 < slope < 2.5:
                    score = length
                    candidates.append((line[0], score))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[1], reverse=True)
        top_lines = [c[0] for c in candidates[:5]]

        avg_x1 = int(np.mean([l[0] for l in top_lines]))
        avg_y1 = int(np.mean([l[1] for l in top_lines]))
        avg_x2 = int(np.mean([l[2] for l in top_lines]))
        avg_y2 = int(np.mean([l[3] for l in top_lines]))

        return avg_x1, avg_y1, avg_x2, avg_y2

    def smooth_line(self, old_line, new_line):
        """
        Exponential moving average.
        """
        if old_line is None:
            return new_line
        a = self.SMOOTH_ALPHA
        return tuple(
            int(a * old + (1 - a) * new)
            for old, new in zip(old_line, new_line)
        )

    def build_info(self, w):
        """
        Returns a dict containing
        - boundary_line: the smoothed line coordinates (or None)
        - road_visible: True if we can see the road boundary
        - boundary_offset: how far the boundary is from the edge
        """
        info = {
            "boundary_line": self.prev_line,
            "road_visible": self.prev_line is not None,
            "boundary_offset": 0,
            "side": self.side,
        }

        if self.prev_line is not None:
            x1, y1, x2, y2 = self.prev_line
            bottom_x = x1 if y1 > y2 else x2

            if self.side == "LEFT":
                info["boundary_offset"] = bottom_x
            else:
                info["boundary_offset"] = w - bottom_x

        return info

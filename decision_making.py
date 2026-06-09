# ============================================================
# decision_making.py — Rule-based driving decisions
# ============================================================

DECISION_COLORS = {
    "STOP":         (0, 0, 255),    # Red
    "SLOW DOWN":    (0, 255, 255),  # Yellow
    "MOVE FORWARD": (0, 255, 0),    # Green
}

def make_decision(detections):
    """Analyze on-road detections and return (decision_str, color)."""

    # Only care about objects actually on the road
    road_objects = [obj for obj in detections if obj.get("on_road") == True]

    # Nothing on the road -> drive
    if not road_objects:
        return "MOVE FORWARD", DECISION_COLORS["MOVE FORWARD"]

    # Priority 1: Person VERY NEAR or NEAR -> STOP
    for obj in road_objects:
        if obj["label"] == "person" and obj["distance"] in ("VERY NEAR", "NEAR"):
            return "STOP", DECISION_COLORS["STOP"]

    # Priority 2: Any object VERY NEAR -> STOP
    for obj in road_objects:
        if obj["distance"] == "VERY NEAR":
            return "STOP", DECISION_COLORS["STOP"]

    # Priority 3: Any object NEAR -> SLOW DOWN
    for obj in road_objects:
        if obj["distance"] == "NEAR":
            return "SLOW DOWN", DECISION_COLORS["SLOW DOWN"]

    # Priority 4: Person FAR -> just slow down a bit (not stop)
    for obj in road_objects:
        if obj["label"] == "person" and obj["distance"] == "FAR":
            return "SLOW DOWN", DECISION_COLORS["SLOW DOWN"]

    # Priority 5: Everything is FAR or VERY FAR -> safe to go
    return "MOVE FORWARD", DECISION_COLORS["MOVE FORWARD"]


def get_steering(lane_info: dict, frame_width: int = 640) -> str:
    """Compute steering direction from lane center offset."""

    lane_center  = lane_info.get("lane_center")
    frame_center = frame_width // 2

    if lane_center is None:
        return "STRAIGHT"

    offset = lane_center - frame_center   # positive = lane is to the right

    MILD_THRESHOLD  = 30
    SHARP_THRESHOLD = 80

    if abs(offset) <= MILD_THRESHOLD:
        return "STRAIGHT"
    elif offset > SHARP_THRESHOLD:
        return "TURN RIGHT"
    elif offset > MILD_THRESHOLD:
        return "SLIGHT RIGHT"
    elif offset < -SHARP_THRESHOLD:
        return "TURN LEFT"
    else:
        return "SLIGHT LEFT"


# ---- Per-stream decision (used for each camera's HUD) ------

def decide(detections, lane_info, frame_width=640, stream_name="FRONT"):
    """
    Return (decision, steering, color) for ONE stream.
    FRONT uses full decision logic. LEFT/RIGHT just report what they see.
    """

    if stream_name == "FRONT":
        # front camera: full decision + lane steering
        decision, color = make_decision(detections)
        steering = get_steering(lane_info, frame_width)
        return decision, steering, color

    # side cameras: report what they see for the HUD
    road_objects = [o for o in detections if o.get("on_road")]

    if any(o["distance"] == "VERY NEAR" for o in road_objects):
        return "BLOCKED", "BLOCKED", (0, 0, 255)        # red
    elif any(o["distance"] == "NEAR" for o in road_objects):
        return "CAUTION", "CAUTION", (0, 200, 255)       # orange
    elif len(road_objects) > 0:
        # objects exist but are FAR — slight steer away
        if stream_name == "LEFT":
            return "OBJ DETECTED", "SLIGHT RIGHT", (0, 220, 180)
        else:
            return "OBJ DETECTED", "SLIGHT LEFT", (0, 220, 180)
    else:
        return "CLEAR", "CLEAR", (0, 255, 0)             # green


# ---- Fused decision — combines all 3 streams ---------------

def fused_decide(front_detections, left_detections, right_detections,
                 front_lane_info, frame_width=640):
    """
    Look at all 3 cameras and make ONE final driving command.
    This is what the car actually follows.

    IMPORTANT: This is distance-based, NOT just object presence.
    Objects that are FAR or VERY FAR are basically ignored.

    Returns:
        (final_decision, final_steering, color)
    """

    # helper: get on-road objects for a stream
    def get_road_objects(detections):
        return [o for o in detections if o.get("on_road")]

    # helper: get only CLOSE objects (NEAR or VERY NEAR)
    def get_close_objects(road_objs):
        return [o for o in road_objs if o["distance"] in ("VERY NEAR", "NEAR")]

    front_road = get_road_objects(front_detections)
    left_road  = get_road_objects(left_detections)
    right_road = get_road_objects(right_detections)

    front_close = get_close_objects(front_road)
    left_close  = get_close_objects(left_road)
    right_close = get_close_objects(right_road)

    # ---- Rule 1: Person CLOSE (NEAR/VERY NEAR) in FRONT = STOP ----
    # (person FAR away is just SLOW DOWN, person VERY FAR is ignored)
    for obj in front_close:
        if obj["label"] == "person":
            return "STOP", "STOP - PERSON", (0, 0, 255)

    # person FAR in front = slow down
    for obj in front_road:
        if obj["label"] == "person" and obj["distance"] == "FAR":
            return "SLOW DOWN", "CAUTION", (0, 255, 255)

    # ---- Rule 2: Check what the FRONT camera sees ----
    front_decision, front_color = make_decision(front_detections)
    steering = get_steering(front_lane_info, frame_width)

    # helper: is a side clear enough to steer into?
    # side is "blocked" only if there are CLOSE objects (NEAR/VERY NEAR)
    def side_is_clear(close_objs):
        return len(close_objs) == 0

    left_ok  = side_is_clear(left_close)
    right_ok = side_is_clear(right_close)

    # Case A: Front is clear
    if front_decision == "MOVE FORWARD":
        # even though front is clear, if a side camera sees objects
        # on the road, we slightly steer away from that side
        left_has_objects  = len(left_road) > 0
        right_has_objects = len(right_road) > 0

        if left_has_objects and not right_has_objects:
            return "MOVE FORWARD", "SLIGHT RIGHT", (0, 255, 0)
        elif right_has_objects and not left_has_objects:
            return "MOVE FORWARD", "SLIGHT LEFT", (0, 255, 0)
        else:
            return "MOVE FORWARD", steering, (0, 255, 0)

    # Case B: Front says STOP (very near object in front)
    if front_decision == "STOP":
        if left_ok and not right_ok:
            return "STEER LEFT", "STEER LEFT", (255, 200, 0)
        elif right_ok and not left_ok:
            return "STEER RIGHT", "STEER RIGHT", (255, 200, 0)
        elif left_ok and right_ok:
            return "STEER LEFT", "STEER LEFT", (255, 200, 0)
        else:
            return "STOP", "STOP", (0, 0, 255)

    # Case C: Front says SLOW DOWN (objects are NEAR)
    if front_decision == "SLOW DOWN":
        if left_ok and not right_ok:
            return "SLOW + LEFT", "SLIGHT LEFT", (0, 200, 255)
        elif right_ok and not left_ok:
            return "SLOW + RIGHT", "SLIGHT RIGHT", (0, 200, 255)
        else:
            return "SLOW DOWN", steering, (0, 255, 255)

    # fallback
    return front_decision, steering, front_color
import cv2
import os
import time
import threading
import queue
import LaneDetection
import object_detection
import decision_making

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_FOLDER = os.path.join(BASE_DIR, "Videos")
VIDEOS = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(".mp4")]

# input_queue    → main sends frames TO yolo thread
# result_lock    → prevents YOLO writing while main is reading
# display_buffer → stores only detections (not YOLO's old frame)
input_queue    = queue.Queue(maxsize=1)
result_lock    = threading.Lock()
display_buffer = {"detections": []}

# yolo worker thread
def yolo_worker():
    """
    Background thread.
    Reads frames from input_queue, runs YOLO detection,
    stores only detections — NOT annotated frame.
    Annotated frame causes flickering so we discard it.
    """
    while True:
        frame = input_queue.get()

        if frame is None:
            print("[Thread] YOLO worker stopped cleanly.")
            break

        detections, _ = object_detection.process_frame(frame, True)

        with result_lock:
            display_buffer["detections"] = detections

def draw_detections(frame, detections):
    """
    Draw bounding boxes on current frame using Member B's colors.
    No scaling needed — YOLO receives exact same size frame as display.
    Colors: red=very near, orange=near, green=far, blue=roadside
    """
    for d in detections:
        x1, y1, x2, y2 = d['box']
        if not d['on_road']:
            draw_color = (255, 0, 0)          # blue — roadside, safe
            thickness  = 2
        else:
            if d['distance'] == 'VERY NEAR':
                draw_color = (0, 0, 255)      # red — stop immediately
            elif d['distance'] == 'NEAR':
                draw_color = (0, 100, 255)    # orange — slow down
            elif d['distance'] == 'FAR':
                draw_color = (0, 200, 100)    # green — safe distance
            else:
                draw_color = (255, 255, 0)    # cyan — very far
            thickness = 3

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, thickness)

        # Draw label
        label  = f"{d['label']} {d['confidence']} | {d['distance']}"
        text_y = max(y1 - 10, 18)
        cv2.putText(frame, label, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, draw_color, 2)

        # Position indicator for on-road objects only
        if d['on_road']:
            cv2.putText(frame, f"[{d['position']}]",
                        (x1, text_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_color, 1)

    return frame

def draw_text_with_bg(img, text, pos, text_color, bg_color):
    """
    Draws text with filled rectangle behind it.
    Keeps text readable over any road background.
    """
    x, y = pos
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x - 5, y - h - 5), (x + w + 5, y + 5), bg_color, -1)
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

def main():
    log_file = open("detection_log.txt", "w")
    log_file.write("frame,fps,lane_center,left_detected,right_detected,decision,steering,on_road_count\n")

    print("\nControls: Q = Quit | N = Next Video | P = Pause")
    print("────────────────────────────────────────────────")

    # Start YOLO background thread once for entire session
    t = threading.Thread(target=yolo_worker, daemon=True)
    t.start()
    print("[Thread] YOLO worker started in background.")

    video_index = 0
    while video_index < len(VIDEOS):
        video_name = VIDEOS[video_index]
        video_path = os.path.join(VIDEO_FOLDER, video_name)

        print(f"\nPlaying [{video_index + 1}/{len(VIDEOS)}]: {video_name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video — skipping: {video_name}")
            video_index += 1
            continue

        frame_count = 0
        prev_time   = time.time()
        fps         = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video finished → next video")
                break

            frame_count += 1
            frame = cv2.resize(frame, (640, 360))

            # LANE DETECTION
            edge_map, lane_frame, lane_info = LaneDetection.process(frame)

            # Only send if thread is free
            if not input_queue.full():
                input_queue.put(frame.copy())

            # Lock so YOLO cannot write while we read
            with result_lock:
                detections = list(display_buffer["detections"])

            # Always start from current frame
            final_frame = frame.copy()

            # Draw lane overlay on current frame
            lane_mask = cv2.cvtColor(lane_frame, cv2.COLOR_BGR2GRAY)
            _, lane_mask = cv2.threshold(lane_mask, 30, 255, cv2.THRESH_BINARY)
            lane_colored = cv2.bitwise_and(
                lane_frame, lane_frame, mask=lane_mask)
            final_frame = cv2.addWeighted(
                final_frame, 1.0, lane_colored, 0.6, 0)

            # Draw YOLO detections on same current frame
            final_frame = draw_detections(final_frame, detections)

            # Decision
            decision_text, steering_text, color = decision_making.decide(
                detections, lane_info, frame.shape[1]
            )

            current_time = time.time()
            new_fps      = 1.0 / max(current_time - prev_time, 0.001)
            fps          = 0.9 * fps + 0.1 * new_fps
            prev_time    = current_time
            on_road  = [d for d in detections if d['on_road']]
            roadside = [d for d in detections if not d['on_road']]

            draw_text_with_bg(final_frame,
                              f"FPS: {fps:.1f}",
                              (10, 20), (0, 255, 255), (0, 0, 0))

            draw_text_with_bg(final_frame,
                              f"Frame: {frame_count}",
                              (10, 45), (255, 255, 0), (0, 0, 0))

            draw_text_with_bg(final_frame,
                              f"Detections: {len(detections)}",
                              (10, 70), (0, 255, 0), (0, 0, 0))

            draw_text_with_bg(final_frame,
                              f"ON ROAD: {len(on_road)}",
                              (10, 95), (0, 0, 255), (0, 0, 0))

            draw_text_with_bg(final_frame,
                              f"Roadside: {len(roadside)}",
                              (10, 120), (255, 0, 255), (0, 0, 0))

            draw_text_with_bg(final_frame,
                              f"Decision: {decision_text}",
                              (10, 170), color, (0, 0, 0))

            draw_text_with_bg(final_frame,
                              f"Steering: {steering_text}",
                              (10, 200), (255, 255, 255), (0, 0, 0))

            # THREAT BANNER
            threat = object_detection.get_nearest_threat(detections)
            if threat:
                banner = (f"{threat['label'].upper()} | "
                          f"{threat['distance']} | "
                          f"{threat['position']}")
                text_size = cv2.getTextSize(
                    banner, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                x = (final_frame.shape[1] - text_size[0]) // 2
                draw_text_with_bg(final_frame, banner,
                                  (x, 40), (0, 0, 255), (0, 0, 0))

            cv2.imshow("Autonomous Car System", final_frame)
            cv2.imshow("Edge Map", edge_map)

            if frame_count % 30 == 0:
                log_file.write(
                    f"{frame_count},{fps:.1f},{lane_info['lane_center']},"
                    f"{lane_info['left_line'] is not None},"
                    f"{lane_info['right_line'] is not None},"
                    f"{decision_text},{steering_text},"
                    f"{len([d for d in detections if d['on_road']])}\n"
                )

            # keys to control
            key = cv2.waitKey(1)
            if key != -1:
                key = chr(key & 0xFF).lower()

                if key == 'q':
                    print("\nQuit by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    input_queue.put(None)
                    return
                elif key == 'n':
                    print("Skipping to next video.")
                    break
                elif key == 'p':
                    print("PAUSED — press P again to resume.")
                    while True:
                        k = cv2.waitKey(0)
                        if chr(k & 0xFF).lower() == 'p':
                            print("RESUMED.")
                            break

        cap.release()
        video_index += 1

    log_file.close()
    cv2.destroyAllWindows()
    input_queue.put(None)

    print("\nAll videos processed. Session complete.")

if __name__ == "__main__":
    main()
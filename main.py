# main.py
"""
Main driver script for the Conveyor Object Detection and Color Counting System.
Integrates camera capture, foreground detection, centroid tracking, color classification,
and line crossing counts with a premium live UI overlay.
"""

import cv2
import time
import config
from camera import ConveyorCamera
from detector import ObjectDetector
from classifier import ColorClassifier
from tracker import CentroidTracker
from counter import ObjectCounter

def main():
    print("=========================================================")
    print("Starting Conveyor Object Detection and Color Counting System")
    print("Press:")
    print("  'q' - Quit")
    print("  'r' - Reset counts")
    print("  'b' - Recalibrate background subtraction model")
    print("=========================================================")

    # Initialize modules
    camera = ConveyorCamera()
    detector = ObjectDetector()
    classifier = ColorClassifier()
    tracker = CentroidTracker()
    counter = ObjectCounter()

    # Start camera capture thread
    if not camera.start():
        print("Fatal Error: Could not initialize camera.")
        return

    # Wait for the camera thread to acquire the first frame
    time.sleep(1.0)

    # Performance monitoring
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0.0

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Copy frame for drawing overlays
            display_frame = frame.copy()

            # 1. Detect objects in the cropped frame
            detections, mask = detector.detect(frame)

            # 2. Update tracking and classification (voting)
            # Passes detector outputs, classifier instance, and cropped frame
            tracked_objects = tracker.update(detections, classifier=classifier, frame=frame)

            # 3. Check for virtual line crossings (Left-to-Right)
            counter.check_crossings(tracker)

            # --- RENDER PREMIUM UI OVERLAYS ---

            # Draw virtual counting line (Red vertical line)
            line_x = config.COUNTING_LINE_X
            h, w = display_frame.shape[:2]
            cv2.line(display_frame, (line_x, 0), (line_x, h), (0, 0, 255), 2)
            cv2.putText(display_frame, "COUNT LINE", (line_x + 5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            # Draw tracked objects bounding boxes and details
            for obj_id, centroid in tracked_objects.items():
                if obj_id in tracker.bboxes:
                    x, y, w_box, h_box = tracker.bboxes[obj_id]
                    cx, cy = centroid
                    obj_class = tracker.get_class(obj_id)

                    # Map classes to distinctive colors for bounding boxes
                    color_map = {
                        "Red Cube": (0, 0, 255),
                        "Blue Cube": (255, 120, 0),
                        "Green Cube": (0, 220, 0),
                        "Yellow Cube": (0, 220, 220),
                        "Orange Cube": (0, 120, 255),
                        "Purple Cube": (200, 0, 200),
                        "Object B": (240, 240, 240),
                        "Unknown Cube": (140, 140, 140)
                    }
                    color = color_map.get(obj_class, (0, 255, 255))

                    # Check if already counted to change indicator
                    is_counted = obj_id in counter.counted_ids
                    thickness = 1 if is_counted else 2

                    # Bounding Box
                    cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), color, thickness)
                    # Centroid Dot
                    cv2.circle(display_frame, (cx, cy), 4, color, -1)
                    
                    # Draw a trailing dot for history to visualize path
                    if obj_id in tracker.history:
                        for p in tracker.history[obj_id][-5:]:
                            cv2.circle(display_frame, p, 2, color, -1)

                    # Label
                    label = f"ID {obj_id} {obj_class}"
                    if is_counted:
                        label += " [COUNTED]"
                    cv2.putText(display_frame, label, (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            # Draw Dark Translucent HUD Background for counts
            hud_w, hud_h = 220, 185
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, display_frame, 0.35, 0, display_frame)

            # Draw Live Counts inside HUD
            counts = counter.get_counts()
            cv2.putText(display_frame, "CONVEYOR COUNTER", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(display_frame, (20, 38), (10 + hud_w - 10, 38), (80, 80, 80), 1)

            categories = [
                ("Red Cubes", counts.get("Red Cube", 0), (0, 0, 255)),
                ("Blue Cubes", counts.get("Blue Cube", 0), (255, 120, 0)),
                ("Green Cubes", counts.get("Green Cube", 0), (0, 220, 0)),
                ("Yellow Cubes", counts.get("Yellow Cube", 0), (0, 220, 220)),
                ("Orange Cubes", counts.get("Orange Cube", 0), (0, 120, 255)),
                ("Purple Cubes", counts.get("Purple Cube", 0), (200, 0, 200))
            ]

            y_pos = 58
            for label, val, text_color in categories:
                # Small color indicator dot
                cv2.circle(display_frame, (28, y_pos - 4), 4, text_color, -1)
                # Count text
                cv2.putText(display_frame, f"{label}: {val}", (40, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                y_pos += 22

            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()

            # Render FPS in the top right
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 80, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Display feed and mask
            cv2.imshow("Conveyor Object Detector", display_frame)
            if mask is not None:
                cv2.imshow("Foreground Mask", mask)

            # Handle Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break
            elif key == ord('r'):
                counter.reset()
            elif key == ord('b'):
                detector.reset_background()

    finally:
        # Stop background camera thread and release window
        camera.stop()
        cv2.destroyAllWindows()
        print("System shut down successfully.")

if __name__ == "__main__":
    main()

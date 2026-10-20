# app.py
"""
Flask Web Application for the Conveyor Object Detection & Counting System.
Serves a mobile-responsive dashboard over the local network.
Runs the computer vision pipeline on a background thread and streams video/counts to the browser.
"""

import cv2
import time
import json
import socket
import threading
from flask import Flask, render_template, Response, jsonify

import config
from camera import ConveyorCamera
from detector import ObjectDetector
from classifier import ColorClassifier
from tracker import CentroidTracker
from counter import ObjectCounter

app = Flask(__name__)

# Global instances
camera = ConveyorCamera()
detector = ObjectDetector()
classifier = ColorClassifier()
tracker = CentroidTracker()
counter = ObjectCounter()

# Thread synchronization
latest_frame_lock = threading.Lock()
latest_jpeg_bytes = None
app_running = True

def get_local_ip():
    """Detect the laptop's local network IP address programmatically."""
    try:
        # Create a dummy socket connection to get routing interface
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback to hostname lookup or localhost
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

def process_conveyor():
    """Background thread running the computer vision pipeline."""
    global latest_jpeg_bytes, app_running
    print("[Vision Thread] Starting CV background processing loop...")

    # Wait for the camera to start up and capture its first frames
    time.sleep(1.0)

    while app_running:
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Create a clean display frame to render overlays
        display_frame = frame.copy()

        # 1. Detect objects
        detections, mask = detector.detect(frame)

        # 2. Update tracker
        tracked_objects = tracker.update(detections, classifier=classifier, frame=frame)

        # 3. Check for virtual line crossings
        counter.check_crossings(tracker)

        # --- DRAW VISUAL OVERLAYS (Matches main.py styling) ---
        
        # Draw vertical counting line (Red)
        line_x = config.COUNTING_LINE_X
        h, w = display_frame.shape[:2]
        cv2.line(display_frame, (line_x, 0), (line_x, h), (0, 0, 255), 2)
        cv2.putText(display_frame, "COUNT LINE", (line_x + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        # Draw bounding boxes and labels for active objects
        for obj_id, centroid in tracked_objects.items():
            if obj_id in tracker.bboxes:
                x, y, w_box, h_box = tracker.bboxes[obj_id]
                cx, cy = centroid
                obj_class = tracker.get_class(obj_id)

                color_map = {
                    "Red Cube": (0, 0, 255),
                    "Blue Cube": (255, 120, 0),
                    "Green Cube": (0, 220, 0),
                    "Yellow Cube": (0, 220, 220),
                    "Orange Cube": (0, 120, 255),
                    "Purple Cube": (200, 0, 200),
                    "Unknown Cube": (140, 140, 140)
                }
                color = color_map.get(obj_class, (0, 255, 255))

                is_counted = obj_id in counter.counted_ids
                thickness = 1 if is_counted else 2

                cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), color, thickness)
                cv2.circle(display_frame, (cx, cy), 4, color, -1)
                
                # History trail
                if obj_id in tracker.history:
                    for p in tracker.history[obj_id][-5:]:
                        cv2.circle(display_frame, p, 2, color, -1)

                label = f"ID {obj_id} {obj_class}"
                if is_counted:
                    label += " [COUNTED]"
                cv2.putText(display_frame, label, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Encode frame to JPEG
        # We compress the quality slightly to 70 for efficient over-the-air transmission
        ret_enc, encoded_img = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ret_enc:
            with latest_frame_lock:
                latest_jpeg_bytes = encoded_img.tobytes()

        # Yield control
        time.sleep(0.015)

    print("[Vision Thread] Stopped CV background processing loop.")

# --- WEB APP ROUTING ---

@app.route('/')
def index():
    """Serve the dashboard page."""
    return render_template('dashboard.html')

def stream_video():
    """Generator for MJPEG video streaming."""
    global latest_jpeg_bytes
    while app_running:
        with latest_frame_lock:
            frame = latest_jpeg_bytes

        if frame is None:
            time.sleep(0.03)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Limit transmission to ~30 FPS
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    """MJPEG video feed endpoint."""
    return Response(stream_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def stream_counts():
    """Generator for Server-Sent Events (SSE) counts updates."""
    last_counts = {}
    while app_running:
        current_counts = counter.get_counts()
        
        # Only yield data if counts have actually updated to save bandwidth
        if current_counts != last_counts:
            last_counts = current_counts.copy()
            # SSE message frame format
            yield f"data: {json.dumps(current_counts)}\n\n"
            
        time.sleep(0.1) # Check changes every 100ms

@app.route('/counts_stream')
def counts_stream():
    """Server-Sent Events endpoint for counts."""
    return Response(stream_counts(), mimetype='text/event-stream')

@app.route('/reset_counts', methods=['POST'])
def reset_counts():
    """API endpoint to clear counting metrics."""
    counter.reset()
    return jsonify({"status": "success", "message": "Counters reset successfully"})

if __name__ == '__main__':
    # 1. Start the camera feed
    if not camera.start():
        print("Fatal Error: Could not initialize camera.")
        exit(1)

    # 2. Spawn the background CV pipeline thread
    cv_thread = threading.Thread(target=process_conveyor)
    cv_thread.daemon = True
    cv_thread.start()

    # 3. Print access instructions in the console
    local_ip = get_local_ip()
    print("\n" + "=" * 60)
    print(" CONVEYOR VISION MONITOR WEB SERVER RUNNING")
    print(f" Access the dashboard on your phone via this URL:")
    print(f"\n      http://{local_ip}:{config.WEB_SERVER_PORT}")
    print("\n IMPORTANT:")
    print(" - Ensure BOTH your laptop and phone are on the same Wi-Fi router.")
    print(" - Press Ctrl+C in this terminal to shut down the server.")
    print("=" * 60 + "\n")

    try:
        # Run Flask Web Server
        # threaded=True handles concurrent requests for video/SSE/HTML
        app.run(host=config.WEB_SERVER_HOST, port=config.WEB_SERVER_PORT, debug=False, threaded=True)
    finally:
        # Stop background threads and clean camera resources on shutdown
        app_running = False
        cv_thread.join(timeout=1.0)
        camera.stop()
        print("Web server and vision pipeline shut down successfully.")

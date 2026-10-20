# camera.py
"""
Camera module for the Conveyor system.
Uses a background thread to read frames from the webcam continuously, reducing latency.
Applies the required top/bottom crop before exposing the frame.
"""

import cv2
import threading
import time
import config

class ConveyorCamera:
    def __init__(self, camera_index=config.CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap = None
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        """Start the frame acquisition thread."""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Set resolution if configured
        if config.FRAME_WIDTH:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        if config.FRAME_HEIGHT:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera with index {self.camera_index}")
            return False

        # Read first frame to initialize
        ret, frame = self.cap.read()
        if ret:
            self.frame = self._crop_frame(frame)
        else:
            print("Warning: Initial frame read failed.")

        self.stopped = False
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        return True

    def _update(self):
        """Continuously capture frames from the camera."""
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            cropped = self._crop_frame(frame)
            with self.lock:
                self.frame = cropped
            
            # Yield control to other threads
            time.sleep(0.005)

    def _crop_frame(self, frame):
        """Crop top and bottom pixels according to config."""
        h, w = frame.shape[:2]
        crop_top = config.CROP_TOP
        crop_bottom = config.CROP_BOTTOM
        
        # Guard against invalid crop values
        if crop_top + crop_bottom >= h:
            print("Error: Crop settings exceed frame height. Returning uncropped frame.")
            return frame
            
        cropped = frame[crop_top : h - crop_bottom, :]
        return cropped

    def read(self):
        """Return the most recently captured cropped frame."""
        with self.lock:
            return self.frame is not None, self.frame

    def stop(self):
        """Stop the thread and release camera resources."""
        self.stopped = True
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        print("Camera thread stopped and resources released.")

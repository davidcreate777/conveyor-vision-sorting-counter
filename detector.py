# detector.py
"""
Object detection module using classical OpenCV techniques.
Uses MOG2 background subtraction and contour analysis to locate objects on the conveyor.
"""

import cv2
import numpy as np
import config

class ObjectDetector:
    def __init__(self):
        # Create MOG2 Background Subtractor
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.MOG2_HISTORY,
            varThreshold=config.MOG2_VAR_THRESHOLD,
            detectShadows=config.MOG2_DETECT_SHADOWS
        )
        
        # Morphological kernels
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    def detect(self, frame):
        """
        Detect objects in the frame.
        Returns:
            - list of detections, each being a dict with:
                'bbox': (x, y, w, h)
                'centroid': (cx, cy)
                'contour': contour points
            - cleaned foreground mask (for display/debugging)
        """
        if frame is None:
            return [], None

        # 1. Apply background subtractor
        fg_mask = self.subtractor.apply(frame)

        # 2. Binary thresholding to eliminate shadows and weak pixels
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # 3. Morphological operations
        # Opening removes small noise particles
        mask_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel_open)
        # Closing fills holes inside the objects
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, self.kernel_close)

        # 4. Contour detection
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.MIN_CONTOUR_AREA:
                continue

            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Centroid
            cx = int(x + w / 2)
            cy = int(y + h / 2)

            detections.append({
                'bbox': (x, y, w, h),
                'centroid': (cx, cy),
                'contour': cnt,
                'area': area
            })

        return detections, mask_cleaned

    def reset_background(self):
        """Reset the background subtractor model (e.g., if conveyor lighting changes)."""
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.MOG2_HISTORY,
            varThreshold=config.MOG2_VAR_THRESHOLD,
            detectShadows=config.MOG2_DETECT_SHADOWS
        )
        print("Background subtractor model reset.")

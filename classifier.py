# classifier.py
"""
Color classification module.
Determines whether an object is White (Object B) or a colored Cube.
If it is a Cube, classifies the dominant color (Red, Blue, Green, Yellow, Orange, Purple) using HSV.
"""

import cv2
import numpy as np
import config

class ColorClassifier:
    def __init__(self):
        # Cache color range arrays for performance
        self.ranges = {}
        for key, val in config.COLOR_RANGES.items():
            lower = np.array(val[:3], dtype=np.uint8)
            upper = np.array(val[3:], dtype=np.uint8)
            self.ranges[key] = (lower, upper)

    def classify(self, frame, bbox):
        """
        Classifies an object inside the bounding box.
        Args:
            frame: Cropped conveyor frame (BGR)
            bbox: (x, y, w, h) bounding box of the object
        Returns:
            String representing class name (e.g., "Object B", "Red Cube", "Blue Cube", etc.)
        """
        x, y, w, h = bbox
        
        # 1. Shrink bounding box slightly to avoid background/border interference
        pad_w = int(w * 0.15)
        pad_h = int(h * 0.15)
        
        y1 = max(0, y + pad_h)
        y2 = min(frame.shape[0], y + h - pad_h)
        x1 = max(0, x + pad_w)
        x2 = min(frame.shape[1], x + w - pad_w)
        
        if (y2 - y1) <= 0 or (x2 - x1) <= 0:
            return "Unknown"

        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Total pixels in the analyzed ROI
        total_pixels = roi.shape[0] * roi.shape[1]

        # 2. Check if the object is White (Object B)
        # White has low saturation and high value
        s_channel = hsv_roi[:, :, 1]
        v_channel = hsv_roi[:, :, 2]
        
        white_pixels = np.sum((s_channel <= config.WHITE_MAX_SATURATION) & (v_channel >= config.WHITE_MIN_VALUE))
        white_ratio = white_pixels / total_pixels

        if white_ratio > 0.4:  # If 40% or more of the object is white, classify as Object B
            return "Object B"

        # 3. If not White, determine dominant cube color
        color_pixel_counts = {}
        
        for color_name, (lower, upper) in self.ranges.items():
            mask = cv2.inRange(hsv_roi, lower, upper)
            count = cv2.countNonZero(mask)
            
            # Combine Red_1 and Red_2
            if color_name.startswith("Red_"):
                base_color = "Red"
            else:
                base_color = color_name
                
            color_pixel_counts[base_color] = color_pixel_counts.get(base_color, 0) + count

        # Find the color with the maximum matching pixels
        dominant_color = None
        max_count = 0
        
        for color, count in color_pixel_counts.items():
            if count > max_count:
                max_count = count
                dominant_color = color

        # Minimum threshold: must have at least 10% of pixels matching the dominant color range
        if dominant_color and (max_count / total_pixels) > 0.1:
            return f"{dominant_color} Cube"
        
        return "Unknown Cube"

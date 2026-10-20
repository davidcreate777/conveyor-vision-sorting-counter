# config.py
"""
Configuration file for the Conveyor Object Detection and Color Counting System.
Contains settings for camera crop, HSV thresholds, detector parameters, tracking, and counting.
"""

# Camera Settings
CAMERA_INDEX = 1
FRAME_WIDTH = 640   # Desired width (None to use default)
FRAME_HEIGHT = 480  # Desired height (None to use default)

# Crop Settings (Applied before any image processing)
CROP_TOP = 0
CROP_BOTTOM = 0

# Object Detection Parameters
MIN_CONTOUR_AREA = 800  # Minimum size (in pixels) of contours to consider
MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 40
MOG2_DETECT_SHADOWS = False

# HSV Color Detection Parameters
# OpenCV HSV range: H [0, 180], S [0, 255], V [0, 255]

# White detection (Object B)
# White is defined by low saturation and high value (brightness).
WHITE_MAX_SATURATION = 60
WHITE_MIN_VALUE = 180

# Cube Color Ranges (H_MIN, S_MIN, V_MIN, H_MAX, S_MAX, V_MAX)
# Since red wraps around 0 and 180, we define two ranges for Red.
COLOR_RANGES = {
    "Red_1": (0, 70, 70, 10, 255, 255),
    "Red_2": (170, 70, 70, 180, 255, 255),
    "Orange": (11, 70, 70, 24, 255, 255),
    "Yellow": (25, 70, 70, 35, 255, 255),
    "Green": (36, 70, 70, 85, 255, 255),
    "Blue": (86, 70, 70, 125, 255, 255),
    "Purple": (126, 70, 70, 169, 255, 255),
}

# Centroid Tracker Parameters
MAX_DISAPPEARED_FRAMES = 10  # Number of frames to wait before forgetting an object
MAX_TRACKING_DISTANCE = 50   # Max pixels an object can move between frames

# Counting Parameters
# Counting line is a vertical line in the cropped coordinate space.
# If cropped frame width is 640, line at 320 is right in the center.
COUNTING_LINE_X = 320

# Web Server Settings
WEB_SERVER_HOST = '0.0.0.0'
WEB_SERVER_PORT = 5000

# Vision-Based Conveyor Object Detection & Color Counting System

An industrial-grade computer vision system designed for conveyor belts to detect moving items, track centroids across frames, classify colors in the HSV color space, and maintain accurate directional tallies, complete with an interactive calibration suite and a Flask web dashboard.

---

## Overview

Industrial sorting and inventory pipelines require low-latency, reliable counting and classification without expensive proprietary hardware. This project implements an automated vision inspection pipeline that:
1. Detects moving objects using background subtraction (MOG2).
2. Tracks objects continuously using a Euclidean distance centroid tracker.
3. Classifies object colors (Red, Orange, Yellow, Green, Blue, Purple, White) using HSV thresholding.
4. Counts objects crossing a calibrated vertical boundary line to prevent duplicate counts.
5. Provides both an OpenCV desktop UI overlay and a Flask web dashboard.

---

## Key Modules & Architecture

- **`detector.py` (Object Detection):** Utilizes `cv2.createBackgroundSubtractorMOG2` with morphological operations (erosion and dilation) to segment moving items and extract clean contours.
- **`tracker.py` (Centroid Tracker):** Implements multi-object tracking by calculating Euclidean distances between existing centroids and newly detected contour centers, with configurable disappearance memory (`MAX_DISAPPEARED_FRAMES = 10`).
- **`classifier.py` (Color Classification):** Converts bounding box regions of interest (ROI) to the HSV color space, masks out noise, and determines color categories based on calibrated saturation, value, and hue ranges (handling dual-range hue wrapping for Red).
- **`counter.py` (Directional Tripwire):** Manages virtual counting line crossings (`COUNTING_LINE_X`) and records per-color totals.
- **`camera.py` (Threaded Capture):** Implements a dedicated background thread for camera frame acquisition to eliminate buffer latency and ensure real-time throughput.
- **`calibrate.py` (HSV Tuner):** Interactive calibration utility with OpenCV trackbars to tune Hue, Saturation, and Value parameters in real-time under varying environmental lighting.
- **`app.py` & `dashboard.html` (Web Dashboard):** Live Flask monitoring dashboard with real-time video streaming, dynamic item counters, and clean telemetry.

---

## Project Structure

```
conveyer/
├── templates/
│   └── dashboard.html                       # Responsive HTML web dashboard
├── app.py                                   # Flask web server and telemetry endpoint
├── calibrate.py                             # Real-time HSV trackbar calibration tool
├── camera.py                                # Threaded camera capture module
├── classifier.py                            # HSV color classifier
├── config.py                                # Central system parameters and thresholds
├── counter.py                               # Line crossing counter module
├── detector.py                              # MOG2 background subtraction detector
├── main.py                                  # Primary desktop CV inspection application
├── requirements.txt                         # Python package dependencies
├── tracker.py                               # Euclidean centroid tracking implementation
├── Vision_Based_Conveyor_System_Report.docx # Detailed project report and architecture documentation
└── .gitignore                               # Git ignore configuration
```

---

## Requirements

### Hardware
- Overhead camera or USB webcam positioned perpendicularly above the conveyor surface.
- Controlled lighting for optimal color discrimination.

### Software
- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - `opencv-python`
  - `numpy`
  - `flask`

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/davidcreate777/conveyor-vision-sorting-counter.git
   cd conveyor-vision-sorting-counter
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Calibrate Color Ranges (Optional but recommended)
To adapt the HSV detector to your specific lighting conditions:
```bash
python calibrate.py
```
- Adjust Hue, Saturation, and Value sliders until the target object is isolated.
- Press `s` in the calibration window to display current thresholds in your terminal.
- Update `config.py` with the calibrated boundaries.

### 2. Run Desktop Inspection Pipeline
Run the main computer vision processing pipeline:
```bash
python main.py
```
- **Controls:**
  - `q` : Quit the application.
  - `r` : Reset current item tallies.
  - `b` : Recalibrate background subtraction model.

### 3. Run Flask Web Dashboard
Run the browser-based monitoring dashboard:
```bash
python app.py
```
- Open your web browser at `http://localhost:5000` to monitor live counts, system status, and video stream.

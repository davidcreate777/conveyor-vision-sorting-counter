# calibrate.py
"""
HSV Calibration tool.
Allows the user to adjust HSV trackbars in real-time to find the optimal color ranges.
Press 's' to save/print the current trackbar values.
Press 'q' to quit.
"""

import cv2
import numpy as np
import config

def nothing(x):
    pass

def main():
    print("====================================================")
    # Print the crop setting used
    print(f"Calibration Tool starting using Camera Index: {config.CAMERA_INDEX}")
    print(f"Frame Crop Settings: Top={config.CROP_TOP}px, Bottom={config.CROP_BOTTOM}px")
    print("Controls:")
    print("  's' : Print current HSV values to terminal")
    print("  'q' : Quit calibration")
    print("====================================================")

    # Initialize VideoCapture
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if config.FRAME_WIDTH:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    if config.FRAME_HEIGHT:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open camera. Try checking the index in config.py")
        return

    # Create window
    cv2.namedWindow("Calibration Panel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration Panel", 600, 350)

    # Create trackbars
    cv2.createTrackbar("Hue Min", "Calibration Panel", 0, 180, nothing)
    cv2.createTrackbar("Hue Max", "Calibration Panel", 180, 180, nothing)
    cv2.createTrackbar("Sat Min", "Calibration Panel", 0, 255, nothing)
    cv2.createTrackbar("Sat Max", "Calibration Panel", 255, 255, nothing)
    cv2.createTrackbar("Val Min", "Calibration Panel", 0, 255, nothing)
    cv2.createTrackbar("Val Max", "Calibration Panel", 255, 255, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        # Crop the frame exactly like the main system
        h, w = frame.shape[:2]
        crop_top = config.CROP_TOP
        crop_bottom = config.CROP_BOTTOM
        if crop_top + crop_bottom < h:
            cropped = frame[crop_top : h - crop_bottom, :]
        else:
            cropped = frame

        # Convert to HSV
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

        # Get trackbar positions
        h_min = cv2.getTrackbarPos("Hue Min", "Calibration Panel")
        h_max = cv2.getTrackbarPos("Hue Max", "Calibration Panel")
        s_min = cv2.getTrackbarPos("Sat Min", "Calibration Panel")
        s_max = cv2.getTrackbarPos("Sat Max", "Calibration Panel")
        v_min = cv2.getTrackbarPos("Val Min", "Calibration Panel")
        v_max = cv2.getTrackbarPos("Val Max", "Calibration Panel")

        # Define bounds and create mask
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Apply mask to cropped frame
        result = cv2.bitwise_and(cropped, cropped, mask=mask)

        # Draw line crossing marker for visualization
        cv2.line(cropped, (config.COUNTING_LINE_X, 0), (config.COUNTING_LINE_X, cropped.shape[0]), (0, 255, 0), 2)
        cv2.putText(cropped, "Virtual Line", (config.COUNTING_LINE_X + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show windows
        cv2.imshow("Cropped Camera Feed (Visual)", cropped)
        cv2.imshow("Filtered Mask (Binary)", mask)
        cv2.imshow("Masked Output (Color)", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"Current Calibration: ({h_min}, {s_min}, {v_min}, {h_max}, {s_max}, {v_max})")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

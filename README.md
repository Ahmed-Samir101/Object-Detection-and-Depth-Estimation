# Chessboard Calibration and YOLO Model Execution

This repository contains tools for camera calibration using a chessboard pattern and running a YOLO model for object detection. Follow the steps below to set up and use the system.

## Requirements
- Python 3.x
- OpenCV (cv2)
- Tablet or device with a camera
- 9×5 chessboard pattern (printed or displayed)

## Setup Instructions

### Option 1: Quick Start with Provided XML
If you want to run the YOLO model directly:
1. Use the provided XML calibration file
2. Run `yoloModel` (no additional setup required)

### Option 2: Full Calibration Process
To perform the complete calibration process:

1. **Chessboard Preparation**
   - Download or create a 9×5 chessboard image on your tablet
   - Measure the actual width of a single square in millimeters

2. **Capture Calibration Images**
   - Run `calibration_images.py`
   - Take 8-10 images of the chessboard from different angles and orientations
     - Left side, right side, tilted, etc.
   - Press 's' to save each image

3. **Perform Stereo Vision Calibration**
   - Run `stereovision_calibration.py`
   - This will generate the calibration parameters needed for accurate measurements

4. **Run YOLO Model**
   - Execute `yoloModel` to use the calibrated system for object detection

## Notes
- Ensure consistent lighting conditions during calibration image capture
- The chessboard should fill most of the image frame for best results
- For accurate results, use the same camera for calibration and detection

## Files
- `calibration_images.py` - Script for capturing calibration images
- `stereovision_calibration.py` - Performs camera calibration
- `yoloModel` - Main object detection model
- Provided XML file - Pre-generated calibration data

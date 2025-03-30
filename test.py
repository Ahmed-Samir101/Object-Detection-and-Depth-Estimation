import cv2
import numpy as np

# Load calibration data
cv_file = cv2.FileStorage("stereo_calibration.xml", cv2.FILE_STORAGE_READ)
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
Q = cv_file.getNode('Q').mat()
cv_file.release()

# Camera specifications
BASELINE = 0.06  # 6cm in meters
PIXEL_SIZE = 0.0014  # 1.4µm typical for mobile sensors
FOCAL_LENGTH_MM = 2.4
FOCAL_LENGTH_PX = (FOCAL_LENGTH_MM / 1000) / PIXEL_SIZE  # ~1714 pixels

# Manual Q matrix override (if calibration is questionable)
Q = np.array([[1, 0, 0, -640/2],
              [0, 1, 0, -480/2],
              [0, 0, 0, FOCAL_LENGTH_PX],
              [0, 0, 1/BASELINE, 0]])

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Stereo matcher configuration
window_size = 11
min_disp = 0
num_disp = 16*12  # 192 disparity levels

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    disp12MaxDiff=5,
    P1=8*3*window_size**2,
    P2=32*3*window_size**2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Create WLS filter
wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
right_matcher = cv2.ximgproc.createRightMatcher(stereo)

# Create trackbars for parameter tuning
cv2.namedWindow('Disparity')
cv2.createTrackbar('numDisparities', 'Disparity', 12, 20, lambda x: None)
cv2.createTrackbar('blockSize', 'Disparity', 11, 21, lambda x: None)
cv2.createTrackbar('uniquenessRatio', 'Disparity', 10, 50, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Split frames
    left_frame = frame[:, :640]
    right_frame = frame[:, 640:]

    # Rectify images
    left_rect = cv2.remap(left_frame, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4)
    right_rect = cv2.remap(right_frame, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4)

    # Convert to grayscale
    gray_left = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    # Update parameters from trackbars
    num_disp = cv2.getTrackbarPos('numDisparities', 'Disparity') * 16
    block_size = cv2.getTrackbarPos('blockSize', 'Disparity')
    uniqueness_ratio = cv2.getTrackbarPos('uniquenessRatio', 'Disparity')
    
    if block_size % 2 == 0:
        block_size += 1
    
    stereo.setNumDisparities(num_disp)
    stereo.setBlockSize(block_size)
    stereo.setUniquenessRatio(uniqueness_ratio)

    # Compute disparity
    disp_left = stereo.compute(gray_left, gray_right).astype(np.float32)/16
    disp_right = right_matcher.compute(gray_right, gray_left).astype(np.float32)/16
    
    # Apply WLS filter
    filtered_disp = wls_filter.filter(disp_left, gray_left, None, disp_right)

    # Calculate depth map
    depth_map = cv2.reprojectImageTo3D(filtered_disp, Q)[:, :, 2]
    depth_map[np.isnan(depth_map)] = 0  # Replace NaN values with 0
    depth_map[np.isinf(depth_map)] = 0  # Replace infinity values with 0
    depth_map[depth_map < 0] = 0       # Replace negative values with 0

    # Visualize results
    disp_vis = cv2.normalize(filtered_disp, None, alpha=0, beta=255, 
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    
    # Add horizontal lines for rectification check
    for y in range(0, 480, 40):
        cv2.line(left_rect, (0, y), (640, y), (0, 255, 0), 1)
        cv2.line(right_rect, (0, y), (640, y), (0, 255, 0), 1)

    # Show depth values
    cv2.putText(left_rect, f"Baseline: {BASELINE*100}cm | Focal: {FOCAL_LENGTH_PX:.0f}px",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # YOLO object detection example
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]  # Ensure proper indexing
    blob = cv2.dnn.blobFromImage(left_rect, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)  # Use fixed output layers

    # Process detections
    for output in outputs:  # Iterate through all outputs
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Get bounding box coordinates
                center_x, center_y, width, height = detection[0:4] * np.array([640, 480, 640, 480])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                w = int(width)
                h = int(height)
                
                # Calculate depth in ROI
                depth_roi = depth_map[max(0, y):min(480, y+h), max(0, x):min(640, x+w)]
                valid_depths = depth_roi[(depth_roi > 200) & (depth_roi < 5000)]  # 20cm-5m
                
                if valid_depths.size > 0:
                    median_depth = np.median(valid_depths)
                    cv2.rectangle(left_rect, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(left_rect, f"{median_depth/1000:.2f}m", 
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Display
    cv2.imshow('Left Rectified', left_rect)
    cv2.imshow('Right Rectified', right_rect)
    cv2.imshow('Disparity', disp_vis)
    cv2.imshow('Depth', (depth_map / 1000).clip(0, 255).astype(np.uint8))  # Convert mm to meters and clip values

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
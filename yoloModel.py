import numpy as np
import cv2 as cv

# Load calibration data
cv_file = cv.FileStorage()
cv_file.open("stereo_calibration.xml", cv.FileStorage_READ)

stereoMapL_x = cv_file.getNode("stereoMapL_x").mat()
stereoMapL_y = cv_file.getNode("stereoMapL_y").mat()
stereoMapR_x = cv_file.getNode("stereoMapR_x").mat()
stereoMapR_y = cv_file.getNode("stereoMapR_y").mat()

# Load YOLO model
net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[int(i) - 1] for i in net.getUnconnectedOutLayers()]

# Camera parameters (adjust based on your calibration)
baseline = 0.06  # 6 cm in meters
focal_length = 800  # Focal length in pixels (example value)

# Initialize stereo matcher
stereo = cv.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=11,
    P1=8*3*11**2,
    P2=32*3*11**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Split and resize images
    left_img = frame[:, :640]
    right_img = frame[:, 640:] 
    # left_img = cv.resize(left_img, (640, 480))
    # right_img = cv.resize(right_img, (640, 480))

    # Rectify images
    left_rect = cv.remap(left_img, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4)
    right_rect = cv.remap(right_img, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4)

    # Compute disparity map
    disparity = stereo.compute(cv.cvtColor(left_rect, cv.COLOR_BGR2GRAY),
                              cv.cvtColor(right_rect, cv.COLOR_BGR2GRAY)).astype(np.float32) / 16.0

    # Object detection
    height, width = left_rect.shape[:2]
    blob = cv.dnn.blobFromImage(left_rect, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression
    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        indexes = indexes.flatten()
    
    # Draw detections with depth
    for i in indexes:
        x, y, w, h = boxes[i]
        try:
            # Get disparity in the detection area
            roi = disparity[y:y+h, x:x+w]
            mask = roi > 0
            if np.any(mask):
                avg_disparity = np.mean(roi[mask])
                if avg_disparity > 0:
                    depth = (baseline * focal_length) / avg_disparity
                    label = f"{classes[class_ids[i]]} {depth:.2f}m"
                else:
                    label = f"{classes[class_ids[i]]} ?m"
            else:
                label = f"{classes[class_ids[i]]} ?m"
            
            # Draw bounding box and label
            color = (0, 255, 0)
            cv.rectangle(left_rect, (x, y), (x+w, y+h), color, 2)
            cv.putText(left_rect, label, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except:
            pass

    cv.imshow("Object Detection", left_rect)
    cv.imshow("Disparity", (disparity - stereo.getMinDisparity()) / stereo.getNumDisparities())

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
import triangulation as tri
import calibration
import mediapipe as mp

mp_facedector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

frame_rate = 30
baseline = 5
focal_length = 2.4
fov = 105

with mp_facedector.FaceDetection(min_detection_confidence=0.2) as face_detector:
    
    while(cap.isOpened()):
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Split the image in half
        left_frame = frame[:, :640]
        right_frame = frame[:, 640:] 

        frame_left, frame_right = calibration.undistortedRectify(left_frame, right_frame)

        if not ret:
            break
        else:
            frame_right = cv.cvtColor(frame_right, cv.COLOR_BGR2RGB)
            frame_left = cv.cvtColor(frame_left, cv.COLOR_BGR2RGB)
            
            # Process the image and find faces 
            results_right= face_detector.process(frame_right)
            results_left= face_detector.process(frame_left)
            # Convert the RGB image to BGR 
            frame_right = cv.cvtColor(frame_right, cv.COLOR_RGB2BGR)
            frame_left = cv.cvtColor(frame_left, cv.COLOR_RGB2BGR)
            

            # calc depth
            center_right = 0
            center_left = 0

            if results_right.detections:
                for id, detection in enumerate(results_right.detections):
                    mp_draw.draw_detection(frame_right, detection)
                    bboxC = detection.location_data.relative_bounding_box

                    h, w, c = frame_right.shape
                    boundBox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                    center_point_right = (int(bboxC.xmin * w + bboxC.width * w / 2), int(bboxC.ymin * h + bboxC.height * h / 2))
                    cv.putText(frame_right, f'{int(detection.score[0] * 100)}%', (boundBox[0], boundBox[1] - 20), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            
            if results_left.detections:
                for id, detection in enumerate(results_left.detections):
                    mp_draw.draw_detection(frame_left, detection)
                    bboxC = detection.location_data.relative_bounding_box

                    h, w, c = frame_left.shape
                    boundBox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                    center_point_left = (int(bboxC.xmin * w + bboxC.width * w / 2), int(bboxC.ymin * h + bboxC.height * h / 2))
                    cv.putText(frame_left, f'{int(detection.score[0] * 100)}%', (boundBox[0], boundBox[1] - 20), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            if not results_left.detections and not results_right.detections:
                cv.putText(frame_left, "No face detected", (50, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                cv.putText(frame_right, "No face detected", (50, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            else:
                depth = tri.find_depth(center_point_left, center_point_right, frame_left, frame_right, baseline, focal_length, fov)
                cv.putText(frame_left, f'Depth: {depth:.2f} cm', (50, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                cv.putText(frame_right, f'Depth: {depth:.2f} cm', (50, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

            end =  time.time()
            total_time = end - start
            fps = 1 / total_time

            cv.putText(frame_left, f'FPS: {fps:.2f}', (50, 100), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            cv.putText(frame_right, f'FPS: {fps:.2f}', (50, 100), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

            cv.imshow("Left Lens", frame_left)
            cv.imshow("Right Lens", frame_right)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break



cap.release()
cv.destroyAllWindows()

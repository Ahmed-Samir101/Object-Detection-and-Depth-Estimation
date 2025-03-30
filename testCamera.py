import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Split the image in half (left and right lenses)
    left_img = frame[:, :640]   # Left half
    right_img = frame[:, 640:]  # Right half

    # Resize for better display
    left_img_resized = cv2.resize(left_img, (640, 480))
    right_img_resized = cv2.resize(right_img, (640, 480))

    # Show both images
    cv2.imshow("Left Lens", left_img_resized)
    cv2.imshow("Right Lens", right_img_resized)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("img/left_image" + str(num) + '.png', left_img_resized)
        cv2.imwrite("img/right_image" + str(num) + '.png', right_img_resized)
        num += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

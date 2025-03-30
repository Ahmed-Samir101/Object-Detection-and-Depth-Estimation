import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

num = 0

while cap.isOpened():
    ret, img = cap.read()

    left_img = img[:,:640]
    right_img = img[:,640:]

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('images/stereo_left/imgL' + str(num) + '.png', left_img)
        cv2.imwrite('images/stereo_right/imgR' + str(num) + '.png', right_img)
        print("Image saved!")
        num += 1

    cv2.imshow("ImgL", left_img)
    cv2.imshow("ImgR", right_img)

cap.release()
cv2.destroyAllWindows()
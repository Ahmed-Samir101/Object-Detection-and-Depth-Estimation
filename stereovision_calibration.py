import numpy as np
import cv2 as cv
import glob

chessboradSize = (9,6)
frameSize = (640,480)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboradSize[0]*chessboradSize[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboradSize[0], 0:chessboradSize[1]].T.reshape(-1,2)

objp *= 19
print(object)

objPoints = [] # 3D
imgPointsL = [] # 2D
imgPointsR = [] # 2D

imagesLift = glob.glob("./images/stereo_left/*.png")
imagesRight = glob.glob("./images/stereo_right/*.png")

for imgLift, imgRight in zip(imagesLift, imagesRight):
    
    imgL = cv.imread(imgLift)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    rectL, cornersL = cv.findChessboardCorners(grayL, chessboradSize, None)
    rectR, cornersR = cv.findChessboardCorners(grayR, chessboradSize, None)

    if rectL and rectR == True:
        
        objPoints.append(objp)
        
        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgPointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgPointsR.append(cornersR)

        # draw
        cv.drawChessboardCorners(imgL, chessboradSize, cornersL, rectL)
        cv.imshow("img left", imgL)
        cv.drawChessboardCorners(imgR, chessboradSize, cornersR, rectR)
        cv.imshow("img right", imgR)
        cv.waitKey(2000)

cv.destroyAllWindows()


# Calibration

retL, cameraMatrixL, distL, revcsL, tvecsL = cv.calibrateCamera(objPoints,imgPointsL,frameSize,None,None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL,distL,(widthL,heightL), 1, (widthL,heightL))

retR, cameraMatrixR, distR, revcsR, tvecsR = cv.calibrateCamera(objPoints,imgPointsR,frameSize,None,None)
heightR, widthR, channelsR = imgL.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR,distR,(widthR,heightR), 1, (widthR,heightR))

# Stereo Calibration

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC

criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)

restStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objPoints, imgPointsL, imgPointsR, cameraMatrixL, distL, cameraMatrixR, distR, frameSize, criteria_stereo, flags)

# Stereo Rect

rectifyScale = 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale, (0, 0))

steroMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
steroMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayL.shape[::-1], cv.CV_16SC2)

print("Saving parameters!")
cv_file = cv.FileStorage("stereo_calibration.xml", cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x', steroMapL[0])
cv_file.write('stereoMapL_y', steroMapL[1])
cv_file.write('stereoMapR_x', steroMapR[0])
cv_file.write('stereoMapR_y', steroMapR[1])
cv_file.write('Q', Q)

cv_file.release()


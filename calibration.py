import numpy as np
import cv2 as cv

cv_file = cv.FileStorage()
cv_file.open("stereo_calibration.xml", cv.FILE_STORAGE_READ)

stereoMapL_x = cv_file.getNode("stereoMapL_x").mat()
stereoMapL_y = cv_file.getNode("stereoMapL_y").mat()
stereoMapR_x = cv_file.getNode("stereoMapR_x").mat()
stereoMapR_y = cv_file.getNode("stereoMapR_y").mat()

def undistortedRectify(imgL, imgR):

    undistortedL = cv.remap(imgL, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    undistortedR = cv.remap(imgR, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

    return undistortedL, undistortedR

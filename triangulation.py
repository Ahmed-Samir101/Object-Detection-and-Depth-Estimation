import numpy as np
import cv2 as cv

def find_depth(leftPoint, rightPoint, frameL, frameR, baseline, focal_length, alpha):
    
    heightL, widthL, depthL = frameL.shape
    heightR, widthR, depthR = frameR.shape

    if widthL == widthR:
        f_pixel = (widthR * 0.5) / np.tan(alpha * 0.5 * np.pi / 180.0)
    else:
        print("Error: Images are not the same size")
    
    x_right = rightPoint[0]
    x_left = leftPoint[0]

    disparity = x_left - x_right

    zDepth = (baseline * f_pixel) / disparity

    return abs(zDepth)


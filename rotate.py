import cv2
import numpy as np

def rotate_image(src,
                angle,
                center):
    H, W, _ = src.shape
    rot_m = cv2.getRotationMatrix2D(center, angle, 1)
    dst = cv2.warpAffine(src.copy(), rot_m, (W, H))
    return dst, rot_m

def rotate_landmark(landmark, rot_m):
    roted_landmark = landmark.copy()
    n = len(landmark) // 3
    for i in range(n):
        pt = [landmark[i*3 + 0], landmark[i*3 + 1], 1]
        rot_pt_x = np.dot(pt, rot_m[0])
        rot_pt_y = np.dot(pt, rot_m[1])
        roted_landmark[i*3 + 0] = rot_pt_x
        roted_landmark[i*3 + 1] = rot_pt_y
        return roted_landmark
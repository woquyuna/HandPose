import numpy as np

class Rectified_Hand_Box:
    def __init__(self, x, y, w, h):
        self.upperLeft_x = x
        self.upperLeft_y = y
        self.w = w
        self.h = h

    def update_with_norm_new_box(self, 
                                anchor_norm, 
                                w_norm, h_norm, 
                                box_size, 
                                r, 
                                minV=0, maxV=0):
        # compute crop into for next frame
        H, W = box_size
        self.w = min(int(w_norm * W * r), W)
        self.h = min(int(h_norm * H * r), H)
        if minV != 0 or maxV != 0:
            self.w = max(min(self.w, maxV), minV)
            self.h = max(min(self.h, maxV), minV)
        self.w = max(self.w, self.h)
        self.h = self.w
        self.upperLeft_x = max(0, int(anchor_norm[0] * W - self.w // 2))
        self.upperLeft_y = max(0, int(anchor_norm[1] * H - self.h // 2))
        # constrain w,h within image bound
        self.w = min(W - self.upperLeft_x, self.w)
        self.h = min(H - self.upperLeft_y, self.h)
    
    def clear(self):
        self.w = 1
        self.h = 1
        self.upperLeft_x = 0
        self.upperLeft_y = 0


'''
* get the new box with current landmark in RECTIFIED CROP HAND
* return the NORMALIZED circumscribed rectangle information
  landmark: [x,y,z] * 21
'''
def get_new_norm_box_info(landmark):
    assert len(landmark) == 21 * 3
    xmin, ymin = 1.0, 1.0
    xmax, ymax = 0.0, 0.0
    n = len(landmark) // 3
    for i in range(n):
        xmin = min(xmin, landmark[i*3 + 0])
        xmax = max(xmax, landmark[i*3 + 0])
        ymin = min(ymin, landmark[i*3 + 1])
        ymax = max(ymax, landmark[i*3 + 1])

    anchor = [(xmin + xmax)/2., (ymin + ymax)/2.]
    w = xmax - xmin
    h = ymax - ymin
    return anchor, w, h


def get_absolute_rect_norm_landmark(norm_landmark,
                                    roted_hand_box,
                                    full_image_size):
    H, W = full_image_size
    roted_landmark = roted_hand_box.copy()
    n = len(norm_landmark) // 3
    for i in range(n):
        roted_landmark[i*3 + 0] = (norm_landmark[i*3 + 0] * roted_hand_box.w + roted_hand_box.upperLeft_x) / W
        roted_landmark[i*3 + 1] = (norm_landmark[i*3 + 1] * roted_hand_box.h + roted_hand_box.upperLeft_y) / H
    return roted_landmark


def predict_rectify_hand_angle(norm_landmark,
                                full_image_size):
    H, W = full_image_size
    palm_bottom_x = int(norm_landmark[3*0+0] * W)
    palm_bottom_y = int(norm_landmark[3*0+1] * H)
    palm_upper_x = int(norm_landmark[3*9+0] * W)
    palm_upper_y = int(norm_landmark[3*9+1] * H)
    delta_angle = (np.arctan2(palm_upper_y - palm_bottom_y, palm_upper_x - palm_bottom_x)) * 180 / np.pi + 90
    return delta_angle
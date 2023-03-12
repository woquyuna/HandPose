import numpy as np
import onnxruntime as ort
import cv2
import copy

from common import Rectified_Hand_Box,get_new_norm_box_info,get_absolute_rect_norm_landmark, predict_rectify_hand_angle
from rotate import rotate_image, rotate_landmark
from draw import draw_landmarks

''' ######
* global constant
    INPUT_SIZE: model input size
'''
INPUT_SIZE = 224

''' ######
* flag
    flag_hand_in_img: flag to judge whether a hand be in the PREVIOUS frame
'''
flag_hand_in_img = False

''' #####
* global variable
    angle_full: angle for rotation in FULL image
    hand_box: rectified(after rotation) hand box
'''
angle_full = 0
roted_hand_box = Rectified_Hand_Box(0,0,1,1)

''' ######
* model
    load hand landmark model
'''
model = ort.InferenceSession("hand_landmark_full.onnx")


''' #####
* camera
    fps:frame rate
'''
fps = 30
cap = cv2.VideoCapture(0)
cap.set(5, fps)


''' ######
* img variable to show when debug
'''
img_r = None
img_hand = None

''' #####
* rotate back landmark and render landmark in source image
    src: source input RGB image
    roted_landmark: absolute landmark in rotated(rectified) full image
    H,W:source image size
'''
def render(src, norm_roted_landmark, angle):
    H, W, _ = src.shape
    # landmark rotation is based on absolute coordinate?
    roted_landmark = norm_roted_landmark.copy()
    n = len(roted_landmark) // 3
    for i in range(n):
        roted_landmark[i*3+0] = norm_roted_landmark[i*3+0] * W
        roted_landmark[i*3+1] = norm_roted_landmark[i*3+1] * H
        '''
        TODO: How to compute rotation back matrix correctly
        rotate landmark back to src image
        center = np.array([W / 2, H / 2, 1])    # image center point to rotate
        # center point SHIFTs because image is a RECANGLE when rotating and then crop to rectangle, square is no need
        center_rot = [0,0]
        center_rot[0] = np.dot((center, rot_m[0]))
        center_rot[1] = np.dot((center, rot_m[1]))
        '''
        # compute rotate back matrix NO NEED FOR TO COMPUTE NEW CENTER
        _, rot_m_inv = rotate_image(src.copy(), -angle, (W/2, H/2))
        landmark = rotate_image(roted_landmark, rot_m_inv)
        # normalize landmark to draw
        n = len(landmark) // 3
        for i in range(n):
            landmark[3*i+0] = min(landmark[3*i+0]/W, 1.0)
            landmark[3*i+1] = min(landmark[3*i+1]/H, 1.0)
        src = draw_landmarks(src, landmark)
        return src

''' ######
main loop
'''
while cap.isOpened():
    print("flag:",flag_hand_in_img)
    print("hand box:{},{},{},{}".format(roted_hand_box.upperLeft_x,
                                        roted_hand_box.upperLeft_y,
                                        roted_hand_box.w,
                                        roted_hand_box.h))
    success, src = cap.read()

    if not success:
        print("Ignoring empty camera frame")
        continue

    # camera RGB frame
    src = cv2.flip(src, 1)  # as a final result to show

    H, W, _ = src.shape

    img = src.copy()

    # judge the input to model
    if flag_hand_in_img:    # A credible hand in previous frame, crop hand area
        img, rot_m = rotate_image(img, angle_full, (W/2, H/2))

        img_r = img.copy()

        img = img[roted_hand_box.upperLeft_y: roted_hand_box.upperLeft_y + roted_hand_box.h,
                  roted_hand_box.upperLeft_x: roted_hand_box.upperLeft_x + roted_hand_box,w,]   # crop rectify hand

        img_hand = img.copy()
    else:   # no hand in previous frame, input full image
        img = img

    # preprocess
    img= cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 255. #(1,224,224,3)

    # run model
    inputs = {model.get_inputs()[0].name: img.transpose(0,3,1,2)}
    results = model.run(None, inputs)

    # model output
    confidence = results[1][0][0]
    norm_landmark = results[0].squeeze()/INPUT_SIZE     # normalize to 0~1

    # a credible hand in image
    if confidence > 0.5:
        if flag_hand_in_img == False:   # first frame to exist a credible hand in iamge
            flag_hand_in_img = True
            src = draw_landmarks(src, norm_landmark)
            anchor_norm, w_norm, h_norm = get_new_norm_box_info(norm_landmark)  # get norm info in full image
            # compute crop info for next frame
            roted_hand_box.update_with_norm_new_box(anchor_norm, w_norm, h_norm, (H,W), 1.2)
        else:
            # TODO:debug, show rectified hand with landmark
            img_hand = draw_landmarks(img_hand, norm_landmark)
            cv2.imshow("Rectified Hand with landmark", img_hand)
            if cv2.waitKey(5) & 0xFF == 'q':
                break
            
            # convert normlized landmark in rectified hand area to absoluate coordinate in ROTATED FULL IMAGE, which is then normalized
            norm_roted_landmark = get_absolute_rect_norm_landmark(norm_landmark, roted_hand_box, (H,W))
            img_r = draw_landmarks(img_r, norm_roted_landmark)

            # TODO:debug rotate full image back
            img_r, _ = rotate_image(img_r, -angle_full, (W/2, H/2))
            cv2.imshow("Rotated back img:", img_r_)
            if cv2.waitKey(5) & 0xFF == 'q':
                break

            # rot landmark back and render in src img
            src = render(src, norm_roted_landmark, angle_full)

            # predict crop info for next frame
            angle_roted = predict_rectify_hand_angle(norm_roted_landmark, (H,W))
            # smooth predicted rotation
            w, w_new = 3, 1
            angle_full_new = angle_full + angle_roted
            angle_full = (angle_full * w + angle_full_new * w_new) / (w + w_new)

            roted_anchor_norm, roted_w_norm, roted_h_norm = get_new_norm_box_info(norm_roted_landmark)
            roted_hand_box.update_with_norm_new_box(roted_anchor_norm, 
                                                    roted_w_norm, 
                                                    roted_h_norm, 
                                                    (H,W), 
                                                    1.5, 
                                                    240, 480)
            # TODO:debug
            cv2.rectangle(img_r,
                          (roted_hand_box.upperLeft_x, roted_hand_box.upperLeft_y),
                          (roted_hand_box.upperLeft_x + roted_hand_box.w, roted_hand_box.upperLeft_y + roted_hand_box.h),
                          (0,0,0), 1)
            cv2.imshow("Rotated full image with box and landmark", img_r)
            if cv2.waitKey(5) & 0xFF == 'q':
                break
    else:
        flag_hand_in_img = False
        roted_hand_box.clear()
        angle_full = 0

    if flag_hand_in_img:
        cv2.circle(src, (10,10), 8, (0,255,0), -1)
    else:
        cv2.circle(src, (10,10), 8, (0,0,255), -1)
    
    cv2.imshow("full image", src)
    if cv2.waitKey(5) & 0xFF == 'q':
        break
    

            



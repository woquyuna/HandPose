import cv2

WHITE_COLOR = (224, 224, 224)
DETECT_SIZE = 192
LANDMARK_SIZE = 224
POINT_COLOR = (0, 0, 255)
LINE_COLOR = (0, 255, 0)

def draw_points(image,
                landmark_list,
                color):
    assert len(landmark_list) == 21*3
    n = len(landmark_list) // 3
    H, W, _ = image.shape
    for i in range(n):
        landmark_px = [int(landmark_list[i*3 + 0] * W), int(landmark_list[i*3 + 1] * H)]
        landmark_px[0] = max(min(landmark_px[0], W), 0)
        landmark_px[1] = max(min(landmark_px[1], H), 0)
        circle_border_radius = max(5 + 1, int(5 * 1.2))
        cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR, -1)
        cv2.circle(image, landmark_px, 5, color, -1)
    return image

def draw_lines(image,
                landmark_list,
                color=LINE_COLOR):
    assert len(landmark_list) == 21*3
    H, W, _ =image.shape
    line_points = [[0,1], [1,2], [2,3], [3,4],
                    [0,5], [5,6], [6,7], [7,8],
                    [5,9], [9,10], [10,11], [11,12],
                    [9,13], [13,14], [14,15], [15,16],
                    [13,17], [17,18], [18,19], [19,20],
                    [0,5], [0,17]]
    
    for line_point in line_points:
        cv2.line(image, 
                (int(landmark_list[line_point[0]*3 + 0]*W), int(landmark_list[line_point[0]*3 + 1]*H)),
                (int(landmark_list[line_point[1]*3 + 0]*W), int(landmark_list[line_point[1]*3 + 1]*H)),
                color)
    return image

def draw_landmarks(image, landmark_list, color=(0,0,255)):
    if landmark_list is None:
        return
    if image.shape[2] != 3:
        raise ValueError("Input image must contain 3 channel bgr data.")
    image = draw_lines(image, landmark_list)
    image = draw_points(image, landmark_list, color)

    return image
import cv2
import os
import numpy as np

def perspective_transform(file_path, box_label):

    img_path = file_path
    filename, ext = os.path.splitext(os.path.basename(img_path))
    ori_img = cv2.imread(img_path)

    # Resize the image for display
    max_display_size = 800
    height, width, _ = ori_img.shape
    if max(height, width) > max_display_size:
        scale_factor = max_display_size / max(height, width)
        ori_img_display = cv2.resize(ori_img, (int(width * scale_factor), int(height * scale_factor)))
    else:
        ori_img_display = ori_img.copy()

    box = list()
    # label_file_path = f'C:/Users/qa762/Desktop/Make_my_profile/yolov5/runs/detect/{file_name}/labels/{self.file_name}.txt'
    # with open(label_file_path, 'r') as file:
    #     text = file.read().strip()
    #
    # box = text.split()
    # print(box)
    # box = [0 ,0.121992, 0.396557, 0.148396, 0.121324, 0.954687]  # [0, xmin, ymin, width, height]
    box = box_label
    yolov5_xmin = max(int((float(box[1]) - float(box[3]) / 2) * ori_img.shape[1]), 0)
    yolov5_ymin = max(int((float(box[2]) - float(box[4]) / 2) * ori_img.shape[0]), 0)
    yolov5_width = int(float(box[3]) * ori_img.shape[1])
    yolov5_height = int(float(box[4]) * ori_img.shape[0])

    # perspective transform
    src = np.array([
        [yolov5_xmin + 3, yolov5_ymin],
        [yolov5_xmin + yolov5_width, yolov5_ymin - 7],
        [yolov5_xmin + yolov5_width - 7, yolov5_ymin + yolov5_height],
        [yolov5_xmin + 3, yolov5_ymin + yolov5_height - 3],
    ], dtype=np.float32)

    dst = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src=src, dst=dst)
    result = cv2.warpPerspective(ori_img, M=M, dsize=(width, height))  # 이 부분 수정

    # Resize the result for display
    if max(height, width) > max_display_size:
        result_display = cv2.resize(result, (int(width * scale_factor), int(height * scale_factor)))
    else:
        result_display = result.copy()

    cv2.imshow('result', result_display)
    cv2.imwrite('result.jpg',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

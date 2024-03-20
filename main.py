import subprocess
import cv2
import perspective_transform
import os
cap = cv2.VideoCapture(0)

frame_count = 0

while True:
    ret, frame = cap.read()  # 웹캠에서 프레임 읽기

    if not ret:
        break

    cv2.imshow('frame', frame)  # 프레임 보여주기
    key = cv2.waitKey(1)
    frame_count += 1

    # 'q' 키를 누르면 종료
    if key & 0xFF == ord('q'):
        file_path = f"capture_{frame_count}.jpg"
        cv2.imwrite(file_path, frame)
        file_name = os.path.basename(file_path)
        file_name_without_extension, _ = os.path.splitext(file_name)
        command = f"python yolov5/detect.py --save-txt --save-conf  --save-crop --name {file_name_without_extension} --weights yolov5/runs/train/barcode_yolov5m_results/weights/barcode_best.onnx --conf 0.5 --source {file_path}"
        result = subprocess.run(command, shell=True, encoding='utf-8')
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()

label_file_path = f'yolov5/runs/detect/{file_name_without_extension}/labels/{file_name_without_extension}.txt'
with open(label_file_path, 'r') as file:
    text = file.read().strip()
box = text.split()
print(box)

c1 = perspective_transform.perspective_transform(file_path,box)



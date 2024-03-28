import cv2
import torch
import numpy as np

path = 'D:/yolov5safetyhelmet-main/yolov5safetyhelmet-main/final.pt'

model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)

#new_class_names = ['helmet', 'boots', 'safety-vests']
#model.names= new_class_names

cap = cv2.VideoCapture("D:\DJI_0141.MP4")
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 600))
    results = model(frame)
    frame = np.squeeze(results.render())
    
    results = model(frame)
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


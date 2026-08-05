import cv2 as cv
import numpy as np
from ultralytics import YOLO

# model_detection = YOLO("models/yolo26n.pt")
model_segmentation = YOLO("models/runs/semantic/train-3/weights/best.pt")
video = cv.VideoCapture('./assets/video/kibukawa_to_minakuchijyonan.mp4')

while True:
    isTrue, frame = video.read()
    frame = frame[540:1060,580:1450]
    output = np.zeros(frame.shape, dtype='uint8')

    results = model_segmentation(frame)

    cv.imshow("YOLO", results[0].plot())

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
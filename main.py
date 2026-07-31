import cv2 as cv
import numpy as np

video = cv.VideoCapture('./img/kibukawa_to_minakucki.mp4')

while True:
    isTrue, frame = video.read()
    frame = frame[540:1060,580:1450]
    output = np.zeros(frame.shape, dtype='uint8')

    # edge detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7,7), 0, borderType=cv.BORDER_REPLICATE)
    canny = cv.Canny(blurred, 87, 255)

    contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for c in contours:
        if len(c) < 87:
            continue
        valid_contours.append(c)

    cv.drawContours(output, valid_contours, -1, (255, 255, 255), 1)

    # curvature calculation

    cv.imshow("video", output)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
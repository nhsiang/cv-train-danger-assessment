import cv2 as cv
import numpy as np

video = cv.VideoCapture('./img/kibukawa_to_minakuchijyonan.mp4')

while True:
    isTrue, frame = video.read()
    frame = frame[540:1060,580:1450]
    output = np.zeros(frame.shape, dtype='uint8')

    # edge detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7,7), 0, borderType=cv.BORDER_REPLICATE)
    canny = cv.Canny(blurred, 87, 255)

    results = []
    lines = cv.HoughLinesP(canny, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=20)
    if lines is not None:
        candidates = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            if abs(abs(angle) - 90) > 40:
                continue
            if length < 40:
                continue

            cv.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow("Hough", output)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
import cv2 as cv
import numpy as np

img = cv.imread('./img/track_first_person_pov.jpg')

# edge detection
blurred = cv.GaussianBlur(img, (7,7), 1)
canny = cv2.Canny(blurred, 150, 150)

cv.imshow('canny', canny)
cv.waitKey(0)
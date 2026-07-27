import cv2
import cv2 as cv

img = cv.imread('./img/track_first_person_pov.jpg')

# edge detection
canny = cv2.Canny(img, 150, 150)

cv.imshow('canny', canny)
cv.waitKey(0)
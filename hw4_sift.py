import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('book.jpg',0)
cap = cv.VideoCapture(0)

sift = cv.xfeatures2d.SIFT_create()

while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(gray, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.45 * n.distance:
            good.append([m])

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, kp1, gray, kp2, good, None, flags=2)

    # Display the resulting frame
    #cv.imshow('book',img1)
    #cv.imshow('frame',gray)
    cv.imshow('matching',img3)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
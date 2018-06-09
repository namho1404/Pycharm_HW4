import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('book.jpg',0)
cap = cv.VideoCapture(0)

orb = cv.ORB_create()

while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(gray, None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv.drawMatches(img1, kp1, gray, kp2, matches[:10],None, flags=2)

    # Display the resulting frame
    #cv.imshow('book',img1)
    #cv.imshow('frame',gray)
    cv.imshow('matching',img3)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
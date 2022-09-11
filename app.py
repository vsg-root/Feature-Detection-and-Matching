import cv2
import numpy as np
import os


orb = cv2.ORB_create(nfeatures=1000)

cap = cv2.VideoCapture(0)

while True:
    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1 = cv2.imread('ImagesQuery/Call of duty ghosts.jpg', 0)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    imgKp1 = cv2.drawKeypoints(img1, kp1, None)
    imgKp2 = cv2.drawKeypoints(img2, kp2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    cv2.imshow('Kp1', imgKp1)
    cv2.imshow('Kp2', imgKp2)
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.imshow('img3', img3)

    cv2.waitKey(1)
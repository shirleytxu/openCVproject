import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)
cv2.waitKey(0)

# image is rotated around point representing 1/3 of width and 1/3 of height
(h, w) = image.shape[:2]
pointMoveAround = (w//3, h//3)

# rotate image by -10 degrees
M = cv2.getRotationMatrix2D(pointMoveAround, -10, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by -10 degrees", rotated)
cv2.waitKey(0)

# rotate image by 30 degrees
rotated = imutils.rotate(image, 30)
cv2.imshow("Rotated by 30 degrees", rotated)
cv2.waitKey(0)
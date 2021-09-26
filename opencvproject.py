import numpy as np
import argparse
import imutils
import cv2

# chapter 3: loading images
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)
cv2.waitKey(0)

# chapter 6: image transformation (rotation and scaling)
# image is rotated around point representing 1/3 of width and 1/3 of height
(h, w) = image.shape[:2]
pointMoveAround = (w//3, h//3)

# rotate image by -10 degrees and scale by 1.7
M = cv2.getRotationMatrix2D(pointMoveAround, -10, 1.7)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by -10 degrees", rotated)
cv2.waitKey(0)

# rotate image by 30 degrees
rotated = imutils.rotate(image, 30)
cv2.imshow("Rotated by 30 degrees", rotated)
cv2.waitKey(0)

# chapter 8: blurring
# gaussian blur with kernel size 9x9
blurred = cv2.GaussianBlur(image, (9, 9), 0)
cv2.imshow("Gaussian", blurred)
cv2.waitKey(0)

# chapter 4: accessing and manipulating pixels
# grabs 400 x 800 pixel region from upper left of image
corner = image[0:450, 0:800]
cv2.imshow("Corner", corner)
cv2.waitKey(0)


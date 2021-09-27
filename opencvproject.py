import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import time

# chapter 3: loading images
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")

# chapter 7: preparing for 3d histogram
ap.add_argument("-s", "--size", required=False, help="Largest color bin size",
                default=5000)
ap.add_argument("-b", "--bins", required=False, help="Num bins per color "
                                                     "channel", default=8)
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)
cv2.waitKey(0)

size = float(args["size"])
bins = int(args["bins"])

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

# chapter 6: image filter
#applies HLS filter
hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS_FULL)
cv2.imshow("HLS", hls)
cv2.waitKey(0)

# chapter 5: drawing shapes
# add "OpenCV" text to center of image
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
fontScale = 5
thickness = 2
white = (255, 255, 255)
pointStart = (w//10, h//2)
cv2.putText(image, 'OpenCV', pointStart, font, fontScale, white, thickness, cv2.LINE_AA)

# add yellow triangle around moon
triangleColor = (76, 174, 206)
pts = [(180, 250), (100, 400), (250, 400)]
cv2.polylines(image, np.array([pts]), True, triangleColor, 5)
cv2.imshow('Triangle Moon', image)
cv2.waitKey(0)

# chapter 7: 3d histogram
hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0,
                                                                   256, 0, 256])
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ratio = size / np.max(hist)

for (x, plane) in enumerate(hist):
    for (y, row) in enumerate(plane):
        for (z, col) in enumerate(row):
            if hist[x][y][z] > 0.0:
                siz = ratio * hist[x][y][z]
                rgb = (z / (bins - 1), y / (bins - 1), x / (bins - 1))
                ax.scatter(x, y, z, s=siz, facecolors=rgb)

plt.show(block=False)

# sets timer for when to close all windows because cv2.waitKey(0) doesn't
# work on the plt window
while True:
    try:
        requestedSecTime = int(input("How many seconds would you like to view "
                                    "these windows for? "))
        break
    except ValueError:
        print("Please enter an integer :)")

while requestedSecTime:
    # divmod returns quotient and remainder of numSecRequested / 60
    numMin, numSec = divmod(requestedSecTime, 60)
    #formats numMin and numSec to two digits, with left digit = 0
    timer = "{:02d}:{:02d}".format(numMin, numSec)
    print(timer, end="\r")
    time.sleep(1)
    requestedSecTime -= 1

print("Closing now!")
plt.close('all')
cv2.destroyAllWindows()
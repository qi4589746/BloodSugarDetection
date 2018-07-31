# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours as contoursTool
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt

# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

# load the example image
image = cv2.imread("p5.jpg")
plt.gray()


# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500)
plt.title('original')
plt.imshow(image)
plt.show()

blurred = cv2.GaussianBlur(image, (7, 7), 0)
plt.title('blurred')
plt.imshow(blurred)
plt.show()

gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
plt.title('gray')
plt.imshow(gray)
plt.show()


edged = cv2.Canny(gray, 30, 150)
plt.title('edged')
plt.imshow(edged)
plt.show()

# find contours in the edge map, then sort them by their
# size in descending order
cntsLevel1 = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                        cv2.CHAIN_APPROX_SIMPLE)
cntsLevel1 = cntsLevel1[0] if imutils.is_cv2() else cntsLevel1[1]
cntsLevel1 = sorted(cntsLevel1, key=cv2.contourArea, reverse=True)
displayCnt = None

imageWithEdges = cv2.drawContours(image.copy(),cntsLevel1,-1,(255,0,255),2)
plt.title('cntsLevel1')
plt.imshow(imageWithEdges)
plt.show()

# loop over the contours
for c in cntsLevel1:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)

    # if the contour has four vertices, then we have found
    # the thermostat display
    if len(approx) == 4:
        displayCnt = approx
        break

#-------------------------------------------------------------------------------

# extract the thermostat display, apply a perspective transform
# to it
warped = four_point_transform(image, displayCnt.reshape(4, 2))
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# output = four_point_transform(image, displayCnt.reshape(4, 2))
(x, y, monitorWight, monitorHigh) = cv2.boundingRect(warped)
halfMonitorHigh = monitorHigh*0.7
oneThirdMonitorHigh = monitorHigh*0.3
plt.title('warped')
plt.imshow(warped)
plt.show()


#convert to gray
thresh = cv2.threshold(warped, 0, 255,	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
thresh = cv2.dilate(thresh,kernel,iterations=1)
plt.imshow(thresh)
plt.show()

edgedKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
edged = cv2.Canny(thresh, 50, 200, 255)
edged = cv2.morphologyEx(edged,cv2.MORPH_CLOSE,edgedKernel)
plt.title("123")
plt.imshow(edged)
plt.show()

# find contours in the thresholded image, then initialize the#
# digit contours lists
cntsLevel2 = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cntsLevel2 = cntsLevel2[0] if imutils.is_cv2() else cntsLevel2[1]

# # loop over the digit area candidates
digitCnts = []
areas = 0
maxY = 0
maxX = 0
minY = 9999
minX = 9999
areaBios = 3

for c in cntsLevel2:
    (ix, iy, iw, ih) = cv2.boundingRect(c)
    if ih > oneThirdMonitorHigh and ih < halfMonitorHigh and iw > 15:
        minY = min(minY, iy - areaBios)
        maxY = max(maxY, iy + ih + areaBios)
        minX = min(minX, ix - areaBios)
        maxX = max(maxX, ix+iw+areaBios)
        areas += cv2.contourArea(c)
        digitCnts.append(c)

thresh2 = thresh[minY:maxY, minX:maxX]
thresh2[:, :areaBios+1] = 0
thresh2[:areaBios+1, :] = 0
thresh2[thresh2.shape[0] - areaBios:, :] = 0
thresh2[:, thresh2.shape[1] - areaBios:] = 0
plt.title('number area')
plt.imshow(thresh2)
plt.show()


(x, y, numberAreaWight, numberAreaHigh) = cv2.boundingRect(thresh2)
halfNumberAreaHigh = numberAreaHigh*0.7
oneThirdNumberAreaHigh = numberAreaHigh*0.3
image, numberConts, numberHierarchy = cv2.findContours(thresh2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

maxNumberW = 0
usedNumberConts = []
for i in range(len(numberConts)):
    (ix, iy, iw, ih) = cv2.boundingRect(numberConts[i])
    if ih > oneThirdNumberAreaHigh and iw > 15 and numberHierarchy[0][i][3] == -1:
        (x, y, w, h) = cv2.boundingRect(numberConts[i])
        maxNumberW = max(w,maxNumberW)
        usedNumberConts.append(numberConts[i])

usedNumberConts = contoursTool.sort_contours(usedNumberConts, method="left-to-right")[0]
numbers = []
thresh3 = cv2.cvtColor(thresh2.copy(), cv2.COLOR_GRAY2BGR)

#-----------------------------------------------------------------------------------------------------
# loop over each of the digits
for c in usedNumberConts:
    # extract the digit ROI
    (x, y, w, h) = cv2.boundingRect(c)
    if w < maxNumberW * 0.5:
        numbers.append(1)
        cv2.rectangle(thresh3, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(thresh3, str(1), (x + 15, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        continue

    roi = thresh2[y:y + h, x:x + w]

    # compute the width and height of each of the 7 segments
    # we are going to examine
    (roiH, roiW) = roi.shape
    (dW, dH) = (int(roiW * 0.2), int(roiH * 0.125))
    dHC = int(dH * 0.5)

    segments = [
        ((2 * dW - dW // 2, 0), (3 * dW + dW // 2, dH - dH // 2)),  # top
        ((dW // 2, dH), (dW + dW //2, 3 * dH)),  # top-left
        ((4 * dW - dW // 2, dH), (w - (dW // 2), 3 * dH)),  # top-right
        ((2 * dW - dW // 2, 4 * dH - dHC), (3 * dW + dW // 2, 4 * dH + dHC)),  # center
        ((dW // 2, 5 * dH), (dW + dW //2, 7 * dH)),  # bottom-left
        ((4 * dW - dW // 2, 5 * dH), (w - (dW // 2), 7 * dH)),  # bottom-right
        ((2 * dW - dW // 2, 7 * dH + dHC), (3 * dW + dW // 2, 8 * dH))  # bottom
    ]
    on = [0] * len(segments)
    # loop over the segments
    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        # extract the segment ROI, count the total number of
        # thresholded pixels in the segment, and then compute
        # the area of the segment
        segROI = roi[yA:yB, xA:xB]
        total = cv2.countNonZero(segROI)
        area = (xB - xA) * (yB - yA)
        # if the total number of non-zero pixels is greater than
        # 50% of the area, mark the segment as "on"
        if total / float(area) > 0.5:
            on[i] = 1

        # lookup the digit and draw it on the image
    digit = DIGITS_LOOKUP[tuple(on)]
    numbers.append(digit)
    cv2.rectangle(thresh3, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(thresh3, str(digit), (x + 15 , y + 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)


# display the digits
print(''.join(map(str, numbers)))
plt.imshow(thresh3)
plt.show()

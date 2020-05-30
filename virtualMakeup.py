# -*- coding: utf-8 -*-

import cv2
import math
import copy
import dlib

import numpy as np

from tkinter import Tk, Button, Label
from tkinter import colorchooser
from PIL import Image, ImageTk


###############################################################################
#
#    GLOBAL VARIABLES
#
###############################################################################

# Init image
g_img = None


# Lips polygon
g_lipsMask = []

# Lips polygon
g_lipsColor = None

# Lips color intensity
g_lipsIntensity = 0.3


# TeethMask
g_teethMask = None

# Teeth are detected or not
g_teethPresence = False

# Teeth whiten strength
g_teethWhitenStrength = 0


# Right Eye
g_eyeROI_R = []
g_eye_R = None

# Left Eye
g_eyeROI_L = []
g_eye_L = None

# Eye color intensity
g_eyesIntensity = 0.15


# Tkinter Label to display the image
g_vLabel = None


###############################################################################
#
#    FACIAL LANDMARKS
#
###############################################################################

# Draw points for any numbers of landmarks models
def renderFacePoints(im, points, roi=(0,0), color=(0, 255, 0), radius=1):
    rx, ry = roi
    for x, y in points:
        cv2.circle(im, (x - rx, y - ry), radius, color, -1)


# Detect facial landmarks in an image
def getLandmarks(faceDetector, landmarkDetector, imRGB, FACE_DOWNSAMPLE_RATIO = 1):
    points = []
    imSmall = cv2.resize(imRGB, None,
                        fx = 1.0 / FACE_DOWNSAMPLE_RATIO,
                        fy = 1.0 / FACE_DOWNSAMPLE_RATIO,
                        interpolation = cv2.INTER_LINEAR)

    faceRects = faceDetector(imSmall, 0)

    if len(faceRects) > 0:
        maxArea = 0
        maxRect = None
        # TODO: Images with multiple faces
        for face in faceRects:
            if face.area() > maxArea:
                maxArea = face.area()
                maxRect = [face.left(),
                           face.top(),
                           face.right(),
                           face.bottom()
                          ]

        rect = dlib.rectangle(*maxRect)
        scaledRect = dlib.rectangle(int(rect.left() * FACE_DOWNSAMPLE_RATIO),
                                    int(rect.top() * FACE_DOWNSAMPLE_RATIO),
                                    int(rect.right() * FACE_DOWNSAMPLE_RATIO),
                                    int(rect.bottom() * FACE_DOWNSAMPLE_RATIO))

        landmarks = landmarkDetector(imRGB, scaledRect)
        points = [(p.x, p.y) for p in landmarks.parts()]
    return points


###############################################################################
#
#    TKINTER CALLBACK
#
###############################################################################


def onColorLipsClick():
    global g_vLabel, g_lipsColor

    g_lipsColor = colorchooser.askcolor(title='Select a lips color')[0]

    applyAll()


def onReinitLipsClick():
    global g_lipsColor

    g_lipsColor = None

    applyAll()


def onColorEyesClick():
    computeGlobalEyesExtractedImages(
            colorchooser.askcolor(title='Select an eye color')[0])

    applyAll()


def onReinitEyesClick():
    initEyesExtractedImages()

    applyAll()


def onWhitenTeethClick():
    global g_teethWhitenStrength

    g_teethWhitenStrength = min(1, g_teethWhitenStrength + 0.05)

    applyAll()


def onReinitTeethClick():
    global g_teethWhitenStrength

    g_teethWhitenStrength = 0

    applyAll()


def onReinitAllClick():
    global g_lipsColor, g_teethWhitenStrength

    g_lipsColor = None
    initEyesExtractedImages()
    g_teethWhitenStrength = 0

    applyAll()


def onSave():
    img = cv2.cvtColor(applyAll(), cv2.COLOR_RGB2BGR)

    cv2.imwrite("image.png", img)


def applyAll():
    global g_vLabel

    img = updateTeeth(g_img)
    img = updateEyes(img)
    img = updateLips(img)

    imgTk = ImageTk.PhotoImage(image=Image.fromarray(img))
    g_vLabel.configure(image=imgTk)
    g_vLabel.image = imgTk

    return img


###############################################################################
#
#    LIPS AND TEETHS
#
###############################################################################

def roiFromPoints(points):
    xmin, ymin = points[0]
    xmax, ymax = points[0]
    for x, y in points:
        xmin = min(x, xmin)
        ymin = min(y, ymin)
        xmax = max(x, xmax)
        ymax = max(y, ymax)
    return (xmin, xmax, ymin, ymax)


def updateLips(img):
    if not g_lipsColor is None:
        imgColor = copy.deepcopy(img)
        for i in range(0, len(imgColor.ravel()), 3):
            if g_lipsMask.ravel()[i]:
                imgColor.ravel()[i:i + 3] = g_lipsColor

        return cv2.addWeighted(
                img, 1 - g_lipsIntensity,
                imgColor, g_lipsIntensity,
                0)
    return img


def updateTeeth(img):
    if g_teethPresence and g_teethWhitenStrength > 0:
        imgColor = copy.deepcopy(img)
        for i in range(0, len(imgColor.ravel()), 3):
            if g_teethMask.ravel()[i]:
                imgColor.ravel()[i:i + 3] = (255, 255, 255)

        return cv2.addWeighted(
                img, 1 - g_teethWhitenStrength,
                imgColor, g_teethWhitenStrength,
                0)
    return img


def createGlobalLipsAndTeethsMasks(points):
    global g_lipsMask, g_teethMask, g_teethPresence

    # Get the polygons from the facial landmarks
    mouthPoly = [points[i] for i in range(48, 60)]
    mouthPoly.append(points[48])
    mouthPoly = np.array([ mouthPoly ], np.int32)

    lipsPolyUp = [points[i] for i in (49, 50, 51, 52, 53, 63, 62, 61, 49)]
    lipsPolyUp = np.array([ lipsPolyUp ], np.int32)

    lipsPolyDown = [points[i] for i in (59, 60, 67, 66, 65, 64, 55, 56, 58, 59)]
    lipsPolyDown = np.array([ lipsPolyDown ], np.int32)

    mouthPolyUnknown = [points[i] for i in range(60, 68)]
    mouthPolyUnknown += [points[60]]
    mouthPolyUnknown = np.array([ mouthPolyUnknown ], np.int32)


    # Get the mouth ROI from the polygon
    roi = roiFromPoints(mouthPoly[0])
    imgRoi = g_img[roi[2]:roi[3], roi[0]:roi[1]]

    # Convert to CIELAB colorspace and take the clarity channel
    clarity, _, _ = cv2.split(cv2.cvtColor(imgRoi, cv2.COLOR_BGR2LAB))


    # Rectify the polygons coord to fit the roi
    for pol in (mouthPoly, lipsPolyUp, lipsPolyDown, mouthPolyUnknown):
        for i in range(len(pol[0])):
            pol[0][i][0] -= roi[0]
            pol[0][i][1] -= roi[2]


    maskMouth = np.zeros(clarity.shape, dtype=np.uint8)
    cv2.fillPoly(
            maskMouth,
            [mouthPoly],
            1)

    maskUnknown = np.zeros(clarity.shape, dtype=np.uint8)
    cv2.fillPoly(
            maskUnknown,
            [mouthPolyUnknown],
            1)

    maskLips = np.zeros(clarity.shape, dtype=np.uint8)
    cv2.fillPoly(
            maskLips,
            [lipsPolyUp],
            1)
    cv2.fillPoly(
            maskLips,
            [lipsPolyDown],
            1)

    # As the teeth usually has a high clarity, compute the lowest clarity
    # value of the lips
    lowerC = 255
    for i, maskedIn in enumerate(maskLips.ravel()):
        if maskedIn:
            lowerC = min(lowerC, clarity.ravel()[i])
    lowerC = (lowerC + 1.5 * 255) / 2.5


    # Create the mask of the teeth
    _, maskTeeth = cv2.threshold(clarity, lowerC, 255, cv2.THRESH_BINARY)
    maskTeeth = maskTeeth / 255
    maskTeeth.ravel()[np.where(maskMouth.ravel() == 0)] = 0
    maskTeeth.ravel()[np.where(maskUnknown.ravel() == 0)] = 0
    maskTeeth.ravel()[np.where(maskLips.ravel())] = 0

    # Check if teeth are found
    # TODO: Find a good threshold
    g_teethPresence = sum(maskTeeth.ravel()) > 0

    # Set the global teeth BGR mask
    g_teethMask = np.zeros(g_img.shape[:2], dtype=np.uint8)
    g_teethMask[roi[2]:roi[3], roi[0]:roi[1]] = maskTeeth
    g_teethMask = cv2.cvtColor(g_teethMask, cv2.COLOR_GRAY2BGR)

    # Set the global lips BGR mask
    g_lipsMask = np.zeros(g_img.shape[:2], dtype=np.uint8)
    if g_teethPresence:
        maskMouth.ravel()[np.where(maskUnknown.ravel())] = 0
    g_lipsMask[roi[2]:roi[3], roi[0]:roi[1]] = maskMouth
    g_lipsMask = cv2.cvtColor(g_lipsMask, cv2.COLOR_GRAY2BGR)


###############################################################################
#
#    EYES
#
###############################################################################

def colorIris(roiEye, color):
    dst = copy.deepcopy(roiEye)

    # Split the RGB chanels
    b, g, r = cv2.split(dst)

    # Compute the sum of the intensities
    sB = sum(sum(b))
    sG = sum(sum(g))
    sR = sum(sum(r))
    maxi = max(sB, max(sG, sR))

    # Select the highest one
    gray = r if maxi == sR else g if maxi == sG else b

    # Reduce the noise
    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    # Threshold
    _, threshold = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)

    # Find the contours on the thresholds
    if (cv2.__version__[0] == '3'):
        _, cnts, _ = cv2.findContours(
                threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        cnts, _ = cv2.findContours(
                threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Keep the longest contour
    cnt = sorted(cnts, key=lambda x:cv2.contourArea(x), reverse=True)[0]

    # Compute circles shape on the blured image
    if (cv2.__version__[0] == '3'):
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1, 20,
                                   param1=50, param2=30,
                                   minRadius=0, maxRadius=0)
    else:
        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=15, param2=30,
                                   minRadius=1, maxRadius=30)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            # Center of the circle
            center = (i[0], i[1])

            # If the center is inside the longest contour
            if cv2.pointPolygonTest(cnt, center, False) == 1:
                # Draw the circle
                cv2.circle(dst, center, i[2], color, -1)

                # Adjust the upper part of the circle to fit with the contours
                xmin, xmax = i[0] - i[2], i[0] + i[2]
                ymin, ymax = i[1] - i[2], int(i[1] - 0.5 * i[2])
                for y in range(ymin, ymax + 1):
                    y2 = (y - i[1])**2
                    for x in range(xmin, xmax + 1):
                        point = (x, y)
                        # If the point (x, y) is either not in the circle or
                        # not strictly inside the contour, remove the added color
                        if math.sqrt(y2 + (x - i[0])**2) > i[2] or \
                           cv2.pointPolygonTest(cnt, point, False) <= 0:
                            dst[y][x] = roiEye[y][x]

                # The circle corresponding to the eye has been treated
                break

    return dst


def createGlobalEyesROI(points, radius=30):
    global g_eyeROI_R, g_eyeROI_L

    # Find the roi for left Eye
    g_eyeROI_L = [
            points[37][0] - radius,
            points[37][1] - radius,
            (points[40][0] - points[37][0] + 2 * radius),
            (points[41][1] - points[37][1] + 2 * radius)
            ]

    # Find the roi for right Eye
    g_eyeROI_R = [
            points[43][0] - radius,
            points[43][1] - radius,
            (points[46][0] - points[43][0] + 2 * radius),
            (points[47][1] - points[43][1] + 2 * radius)
            ]


def computeGlobalEyesExtractedImages(color):
    global g_eye_R, g_eye_L

    initEyesExtractedImages()

    g_eye_R = colorIris(g_eye_R, color)
    g_eye_L = colorIris(g_eye_L, color)


def initEyesExtractedImages():
    global g_eye_R, g_eye_L

    # Extracted image of the left eye
    g_eye_L = g_img[g_eyeROI_L[1]:g_eyeROI_L[1] + g_eyeROI_L[3],
                  g_eyeROI_L[0]:g_eyeROI_L[0] + g_eyeROI_L[2]]

    # Extracted image of the right eye
    g_eye_R = g_img[g_eyeROI_R[1]:g_eyeROI_R[1] + g_eyeROI_R[3],
                  g_eyeROI_R[0]:g_eyeROI_R[0] + g_eyeROI_R[2]]


def updateEyes(img):
    imgEyes = copy.deepcopy(img)

    imgEyes[g_eyeROI_L[1]:g_eyeROI_L[1] + g_eyeROI_L[3],
           g_eyeROI_L[0]:g_eyeROI_L[0] + g_eyeROI_L[2]] = g_eye_L
    imgEyes[g_eyeROI_R[1]:g_eyeROI_R[1] + g_eyeROI_R[3],
           g_eyeROI_R[0]:g_eyeROI_R[0] + g_eyeROI_R[2]] = g_eye_R

    return cv2.addWeighted(img, 1 - g_eyesIntensity,
                           imgEyes, g_eyesIntensity,
                           0)


###############################################################################
#
#    MAIN
#
###############################################################################


path1 = "data/images/girl-no-makeup.jpg"
path2 = "data/images/face1.png"
path3 = "data/images/face2.jpg"

def main():
    global g_img, g_vLabel

    # Input image in RGB format
    g_img = cv2.cvtColor(
                cv2.imread(path1),
                cv2.COLOR_BGR2RGB)

    # Landmark model location
    PREDICTOR_PATH = "data/models/shape_predictor_68_face_landmarks.dat"

    # Get the face detector
    faceDetector = dlib.get_frontal_face_detector()

    # The landmark detector is implemented in the shape_predictor class
    landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

    # Get the facial landmark points
    points = getLandmarks(faceDetector, landmarkDetector, g_img)

    # Create Lips and polygon masks
    createGlobalLipsAndTeethsMasks(points)

    # Create the eyes ROI
    createGlobalEyesROI(points)
    initEyesExtractedImages()

    # Tkinter
    root = Tk()

    # Button to change the color of the lips
    buttonLipsColor = Button(
            root,
            text='Change lips color',
            command=onColorLipsClick)
    buttonLipsColor.grid(row=0, column=2)

    # Button to reinit the color of the lips
    buttonReinitLips = Button(
            root,
            text='Reinit lips color',
            command=onReinitLipsClick)
    buttonReinitLips.grid(row=1, column=2)


    # Button to change the color of the eyes
    buttonLipsColor = Button(
            root,
            text='Change eyes color',
            command=onColorEyesClick)
    buttonLipsColor.grid(row=0, column=4)

    # Button to reinit the color of the eyes
    buttonReinitLips = Button(
            root,
            text='Reinit eyes color',
            command=onReinitEyesClick)
    buttonReinitLips.grid(row=1, column=4)


    # Button to change the color of the eyes
    buttonLipsColor = Button(
            root,
            text='Whiten teeth',
            command=onWhitenTeethClick)
    buttonLipsColor.grid(row=0, column=6)

    # Button to reinit the color of the eyes
    buttonReinitLips = Button(
            root,
            text='Reinit teeth',
            command=onReinitTeethClick)
    buttonReinitLips.grid(row=1, column=6)


    # Button to reinit all
    buttonReinitLips = Button(
            root,
            text='Reinit all',
            command=onReinitAllClick)
    buttonReinitLips.grid(row=0, column=8)

    # Button to save as
    buttonReinitLips = Button(
            root,
            text='Save',
            command=onSave)
    buttonReinitLips.grid(row=1, column=8)


    # Image to display
    imgTk = ImageTk.PhotoImage(image=Image.fromarray(g_img))
    g_vLabel = Label(root, image=imgTk)
    g_vLabel.grid(row=2, column=0, columnspan=10)


    # Launch the app
    root.mainloop()

if __name__ == '__main__':
    main()
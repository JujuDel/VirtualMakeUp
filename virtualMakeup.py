# -*- coding: utf-8 -*-

import cv2
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
g_lipsMask = None

# Lips color
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
g_eyesMask = None

# Eyes color
g_eyesColor = None

# Eyes color intensity
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


def roiFromPoints(points):
    xmin, ymin = points[0]
    xmax, ymax = points[0]
    for x, y in points:
        xmin = min(x, xmin)
        ymin = min(y, ymin)
        xmax = max(x, xmax)
        ymax = max(y, ymax)
    return [xmin, xmax, ymin, ymax]


###############################################################################
#
#    TKINTER CALLBACK
#
###############################################################################


def onColorLipsClick():
    global g_lipsColor

    g_lipsColor = colorchooser.askcolor(title='Select a lips color')[0]

    applyAll()


def onReinitLipsClick():
    global g_lipsColor

    g_lipsColor = None

    applyAll()


def onColorEyesClick():
    global g_eyesColor

    g_eyesColor = colorchooser.askcolor(title='Select an eye color')[0]

    applyAll()


def onReinitEyesClick():
    global g_eyesColor

    g_eyesColor = None

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
    global g_lipsColor, g_eyesColor, g_teethWhitenStrength

    g_lipsColor = None
    g_eyesColor = None
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

def computeIrisMask(imgEye, roiEye, points):
    poly = [[x, y] for x, y in points]

    for i in range(len(poly)):
        poly[i][0] -= roiEye[0]
        poly[i][1] -= roiEye[1]

    poly.append(poly[0])
    poly = np.array([ poly ], np.int32)

    maskEye = np.zeros(imgEye.shape[:2], dtype=imgEye.dtype)
    cv2.fillPoly(
        maskEye,
        [poly],
        1)

    # Keep the blue channel
    gray, _, _ = cv2.split(imgEye)
    gray.ravel()[np.where(maskEye.ravel() == 0)] = 0

    # Compute value for thresholding
    average = 1. * sum(gray.ravel()) / sum(maskEye.ravel())
    average = (255. + 3 * average) / 4.

    # Reduce the noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold
    _, threshold = cv2.threshold(gray, average, 255, cv2.THRESH_BINARY_INV)
    threshold.ravel()[np.where(maskEye.ravel() == 0)] = 0

    # Keep the iris only
    threshold = cv2.bitwise_and(threshold, maskEye * 255)

    # Erode and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    threshold = cv2.erode(threshold, kernel)
    threshold = cv2.dilate(threshold, kernel)

    # Resulting mask
    res = np.zeros(imgEye.shape[:2], dtype=imgEye.dtype)
    res.ravel()[np.where(threshold.ravel() != 0)] = 1

    return res


def createGlobalEyesMask(points, radius=30):
    global g_eyesMask

    # Find the roi for left Eye
    eyeROI_L = roiFromPoints(points[36:42])
    eyeROI_L[2] -= (eyeROI_L[3] - eyeROI_L[2]) // 2
    eyeROI_L[3] += (eyeROI_L[3] - eyeROI_L[2]) // 2

    # Find the roi for right Eye
    eyeROI_R = roiFromPoints(points[42:48])
    eyeROI_R[2] -= (eyeROI_R[3] - eyeROI_R[2]) // 2
    eyeROI_R[3] += (eyeROI_R[3] - eyeROI_R[2]) // 2

    # Extract roi of the left eye
    eye_L = g_img[eyeROI_L[2]:eyeROI_L[3],
                  eyeROI_L[0]:eyeROI_L[1]]

    # Extract roi of the right eye
    eye_R = g_img[eyeROI_R[2]:eyeROI_R[3],
                  eyeROI_R[0]:eyeROI_R[1]]

    # Compute the mask per eyes
    maskL = computeIrisMask(eye_L, (eyeROI_L[0], eyeROI_L[2]), points[36:42])
    maskL = cv2.cvtColor(maskL, cv2.COLOR_GRAY2BGR)
    maskR = computeIrisMask(eye_R, (eyeROI_R[0], eyeROI_R[2]), points[42:48])
    maskR = cv2.cvtColor(maskR, cv2.COLOR_GRAY2BGR)

    # Combine both masks
    g_eyesMask = np.zeros(g_img.shape, dtype=g_img.dtype)
    g_eyesMask[eyeROI_R[2]:eyeROI_R[3],
               eyeROI_R[0]:eyeROI_R[1]] = maskR
    g_eyesMask[eyeROI_L[2]:eyeROI_L[3],
               eyeROI_L[0]:eyeROI_L[1]] = maskL


def updateEyes(img):
    if not g_eyesColor is None:
        imgColor = copy.deepcopy(img)
        for i in range(0, len(imgColor.ravel()), 3):
            if g_eyesMask.ravel()[i]:
                imgColor.ravel()[i:i + 3] = g_eyesColor

        return cv2.addWeighted(
                img, 1 - g_eyesIntensity,
                imgColor, g_eyesIntensity,
                0)
    return img


###############################################################################
#
#    MAIN
#
###############################################################################


path1 = "data/images/girl-no-makeup.jpg"
path2 = "data/images/face1.png"
path3 = "data/images/face2.png"


def main():
    global g_img, g_vLabel

    # Input image in RGB format
    g_img = cv2.cvtColor(
                cv2.imread(path3),
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
    createGlobalEyesMask(points)

    # Tkinter
    root = Tk()
    root.title("VirtualMakeUp")

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
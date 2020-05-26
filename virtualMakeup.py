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

# Init image
g_currentImg = None

# Lips polygon
g_lipsPolygon = []

# Lips polygon
g_lipsColor = None

# Lips color intensity
g_lipsIntensity = 0.3

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
def renderFacePoints(im, points, color=(0, 255, 0), radius=1):
    for x, y in points:
        cv2.circle(im, (x, y), radius, color, -1)


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

    imLips = copy.deepcopy(g_img)

    imLips = updateEyes(imLips)
    imLips = updateLips(imLips)

    imgTk = ImageTk.PhotoImage(image=Image.fromarray(imLips))
    g_vLabel.configure(image=imgTk)
    g_vLabel.image = imgTk


def onReinitLipsClick():
    global g_vLabel, g_lipsColor

    g_lipsColor = None

    imEyes = copy.deepcopy(g_img)
    imEyes = updateEyes(imEyes)

    imgTk = ImageTk.PhotoImage(image=Image.fromarray(imEyes))
    g_vLabel.configure(image=imgTk)
    g_vLabel.image = imgTk


def onColorEyesClick():
    global g_vLabel

    computeGlobalEyesExtractedImages(
            colorchooser.askcolor(title='Select an eye color')[0])

    imEyes = copy.deepcopy(g_img)

    imEyes = updateLips(imEyes)
    imEyes = updateEyes(imEyes)

    imgTk = ImageTk.PhotoImage(image=Image.fromarray(imEyes))
    g_vLabel.configure(image=imgTk)
    g_vLabel.image = imgTk


def onReinitEyesClick():
    global g_vLabel

    initEyesExtractedImages()

    imLips = copy.deepcopy(g_img)
    imLips = updateLips(imLips)

    imgTk = ImageTk.PhotoImage(image=Image.fromarray(imLips))
    g_vLabel.configure(image=imgTk)
    g_vLabel.image = imgTk


def onReinitAllClick():
    global g_vLabel, g_lipsColor

    g_lipsColor = None
    initEyesExtractedImages()

    imgTk = ImageTk.PhotoImage(image=Image.fromarray(g_img))
    g_vLabel.configure(image=imgTk)
    g_vLabel.image = imgTk

###############################################################################
#
#    LIPS
#
###############################################################################

def createLipsPolygon(points):
    global g_lipsPolygon

    g_lipsPolygon = [points[i] for i in range(48, 60)]
    g_lipsPolygon.append(points[48])
    g_lipsPolygon = np.array([ g_lipsPolygon ], np.int32)


def updateLips(img):
    if not g_lipsColor is None:
        imgLips = copy.deepcopy(g_img)

        cv2.fillPoly(
            imgLips,
            [g_lipsPolygon],
            g_lipsColor)

        return cv2.addWeighted(
                img, 1 - g_lipsIntensity,
                imgLips, g_lipsIntensity,
                0)
    return img

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

def main():
    global g_img, g_vLabel

    # Input image in RGB format
    g_img = cv2.cvtColor(
                cv2.imread("data/images/girl-no-makeup.jpg"),
                cv2.COLOR_BGR2RGB)

    # Landmark model location
    PREDICTOR_PATH = "data/models/shape_predictor_68_face_landmarks.dat"

    # Get the face detector
    faceDetector = dlib.get_frontal_face_detector()

    # The landmark detector is implemented in the shape_predictor class
    landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

    # Get the facial landmark points
    points = getLandmarks(faceDetector, landmarkDetector, g_img)

    # Create the lips polygon
    createLipsPolygon(points)

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
    buttonReinitLips.grid(row=0, column=3)

    # Button to reinit all
    buttonReinitLips = Button(
            root,
            text='Reinit all',
            command=onReinitAllClick)
    buttonReinitLips.grid(row=0, column=6)

    # Button to change the color of the eyes
    buttonLipsColor = Button(
            root,
            text='Change eyes color',
            command=onColorEyesClick)
    buttonLipsColor.grid(row=0, column=9)

    # Button to reinit the color of the eyes
    buttonReinitLips = Button(
            root,
            text='Reinit eyes color',
            command=onReinitEyesClick)
    buttonReinitLips.grid(row=0, column=10)

    # Image to display
    imgTk = ImageTk.PhotoImage(image=Image.fromarray(g_img))
    g_vLabel = Label(root, image=imgTk)
    g_vLabel.grid(row=1, column=0, columnspan=15)

    # Launch the app
    root.mainloop()

if __name__ == '__main__':
    main()
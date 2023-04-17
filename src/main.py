import numpy as np
from imutils.video import VideoStream
import argparse
import cv2
import imutils
import time
from config_parser import load_config

config = load_config("config")

# redL = (160,50,50)
# redU = (180,255,255) are the original values range for Red Colour (172,100,100), (179,255,255)
redL = np.array([172, 100, 100])
# (120,70,70) updated with suitable values
redU = np.array([180, 255, 255])
vs = cv2.VideoCapture(config['path_to_file'])
while True:
    frame = vs.read()
    if frame is None:
        break
    frame = frame[1] if frame is not None else frame
    # handle the frame from VideoCapture or VideoStream
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    # blurred = cv2.medianBlur(frame, 1)
    # sharp = cv2.addWeighted(frame, 7.5, blurred, -6.5, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "red", then perform
    # a series of dilation followed by erosion to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, redL, redU)  # , cv2.COLOR_HSV2RGB)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)

    res = cv2.bitwise_and(frame, frame, mask=mask)
    # find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # only proceed if at least one contour was found
    if len(cnts) > 1:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle on the frame,
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    # if the 'Esc' key is pressed, stop the loop
    key = cv2.waitKey(0) & 0xFF
    if key > 1:
        break
vs.release()
# close all windows
cv2.destroyAllWindows()

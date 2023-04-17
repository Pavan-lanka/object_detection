from imutils.video import VideoStream
import argparse
import cv2
import imutils
import time
#constructing argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video")
args = vars(ap.parse_args())
#redL = (160,50,50)
#redU = (180,255,255) are the original values range for Red Colour (172,100,100), (179,255,255)
redL = (172,100,100) # (120,70,70) updated with suitable values
redU = (180,255,255)
# if no video is provided opens video source(i/e Webcam)
if not args.get("video", False):
    vs = VideoStream(src=0).start()
# else, a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])
#allow the camera or video to warm up
time.sleep(2.0)
while True:
    frame = vs.read()
    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break
    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (1, 1), 0)
    sharp = cv2.addWeighted(frame, 7.5, blurred, -6.5, 0)
    hsv = cv2.cvtColor(sharp, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "red", then perform
    # a series of dilation followed by erosion to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, redL, redU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # only proceed if at least one contour was found
    if len(cnts) > 2:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        # only proceed if the radius meets a minimum size
        if radius > 25:
            # draw the circle on the frame,
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    # if the 'Esc' key is pressed, stop the loop
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()
# otherwise, release the camera
else:
    vs.release()
# close all windows
cv2.destroyAllWindows()
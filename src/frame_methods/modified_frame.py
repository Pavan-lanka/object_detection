import cv2 as cv
import imutils


def add_artifacts(video_frame, lower_bound, upper_bound):
    frame = imutils.resize(video_frame, width=600)
    blur_added_frame = cv.GaussianBlur(frame, (11, 11), 0)
    hsv = cv.cvtColor(blur_added_frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_bound, upper_bound, cv.COLOR_HSV2RGB)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, None)
    return mask


def find_contours_to_circle(masked_frame, vid_frame):
    contours = cv.findContours(masked_frame.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if len(contours) >= 1:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle
        circle_vals = max(contours, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(circle_vals)
        # only proceed if the radius meets a minimum size
        if radius > 5:
            # draw the circle on the frame,
            cv.circle(vid_frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv.imshow("Detect_Circle", vid_frame)
    return None




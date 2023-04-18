import cv2 as cv
import imutils


def adding_artifact(video_frame):
    blur_added_frame = cv.GaussianBlur(video_frame, (11, 11), 0)
    return blur_added_frame


def adding_mask(blurred_frame, color_lower_bound, color_upper_bound):
    hsv = cv.cvtColor(blurred_frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, color_lower_bound, color_upper_bound, cv.COLOR_HSV2RGB)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, None)
    return mask


def find_contours(masked_frame):
    contours = cv.findContours(masked_frame.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return contours


def draw_circle(contours, vid_frame):

    if len(contours) >= 1:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle
        circle_vals = max(contours, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(circle_vals)
        # only proceed if the radius meets a minimum size
        if radius > 5:
            # draw the circle on the frame,
            cv.circle(vid_frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
    return vid_frame




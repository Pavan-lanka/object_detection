from src.config_parser import load_config
import cv2 as cv
import imutils

config = load_config("config")


def read_inp_into_frame(video_path):
    vid_src = cv.VideoCapture(video_path)
    while True:
        frame = vid_src.read()
        frame = frame[1] if frame is not None else frame
        if frame is None:
            break
        else:
            return frame


def add_artifacts(video_frame):
    frame = imutils.resize(video_frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)


def color_subtraction(frame_artifacts, lower_bound, upper_bound):
    """
    This method converts the detected frame into HSV and creates a mask to the frame to detect contour
    Args:
        frame_artifacts(numpy.ndarray):  A Frame of video to convert into HSV Color-space
        lower_bound(tuple): A tuple containing Lower boundary of the selected Color
        upper_bound(tuple): A tuple containing Upper boundary of the selected Color

    Returns(numpy.ndarray): A Masked Frame converted into RGB color space
    """
    hsv = cv.cvtColor(frame_artifacts, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_bound, upper_bound, cv.COLOR_HSV2RGB)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, None)
    return mask


def find_contours(masked_frame):
    cnts = cv.findContours(masked_frame.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts


def draw_circle(contours, vid_frame):
    if len(contours) > 1:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle
        c = max(contours, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(c)
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle on the frame,
            cv.circle(vid_frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
    return vid_frame

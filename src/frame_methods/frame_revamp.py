import cv2 as cv
import imutils
import numpy as np


def adding_artifact(video_frame):
    """
        This method accepts a video_frame and adds Gaussian Blur to it.
    Args:
        video_frame(numpy.ndarray): Parameter accepts video frame as a numpy array as input argument

    Returns(numpy.ndarray):
        Parameter returns a numpy array as input argument

    """
    blur_added_frame = cv.GaussianBlur(video_frame, (11, 11), 7)
    return blur_added_frame


def adding_mask(blurred_frame, color_lower_bound, color_upper_bound):
    """
        This method accepts an artifact added video_frame and creates a mask to it.
    Args:
        blurred_frame(numpy.ndarray): Parameter accepts video frame as a numpy array as input argument
        color_lower_bound(tuple): Parameter accepts a tuple of color bounds as integers
        color_upper_bound(tuple): Parameter accepts a tuple of color bounds as integers

    Returns(numpy.ndarray): Returns the masked frame as an array

    """
    hsv = cv.cvtColor(blurred_frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, color_lower_bound, color_upper_bound, cv.COLOR_HSV2RGB)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, None)
    return mask


def find_contours(masked_frame):
    """
        This Method finds the contours for specified color in the input video frame and returns the tuple of
        co-ordinates.
    Args:
        masked_frame(numpy.ndarray): Parameter accepts input argument as a video frame

    Returns(tuple): A tuple of detected contour co-ordinates in the frame

    """
    contours = cv.findContours(masked_frame.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return contours


def draw_circle(contours, vid_frame):
    """
        This method takes the contours and create a circular shape around the detected contours
    Args:
        contours(tuple): A tuple of detected contour co-ordinates in the frame:
        vid_frame(numpy.ndarray): Parameter accepts video frame as a numpy array as input argument

    Returns(numpy.ndarray): Returns the video frame .

    """
    if len(contours) >= 1:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle
        circle_vals = max(contours, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(circle_vals)
        if radius > 5:
            # draw the circle on the frame,
            cv.circle(vid_frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
    return vid_frame


# def salt_pepper(img, percent: int, slt_ppr_percent: tuple = (50, 50)):
#     img = cv.imread(img)
#     slt_ratio = int((slt_ppr_percent[0] / 100) * percent)
#     pep_ratio = int((slt_ppr_percent[1] / 100) * percent)
#     pixels = img.shape[0] * img.shape[1] if len(img.shape) >= 3 else 0
#     if percent <= 100 and pixels > 0:
#         percent = int((percent / 100) * pixels)
#     else:
#         raise Exception('Select valid Percentage to add the noise 0 -- 100')
#     img_copy = img.copy()
#     pix_coordinates = list()
#     while len(pix_coordinates) < percent:
#         x = np.random.randint(0, img.shape[0])
#         y = np.random.randint(0, img.shape[1])
#         cord = (x, y)
#         if cord not in pix_coordinates:
#             pix_coordinates.append(cord)
#     if slt_ppr_percent[0] + slt_ppr_percent[1] > 100:
#         raise ValueError('Give Valid Ration for salt and pepper')
#     for layer in range(img.shape[2]):
#         for tup in pix_coordinates:
from config_parser import load_config
from frame_methods import frame_revamp as fr
import cv2 as cv
import imutils
import time
import numpy as np

config = load_config("config")


def main():
    vid_src = cv.VideoCapture(config['path_to_file'])
    time.sleep(2.0)
    while True:
        ret, frame = vid_src.read()
        if not ret or frame is None:
            break
        frame = imutils.resize(frame, height=600, width=600)
        blurred_frame = fr.adding_artifact(frame)
        masked_frame = fr.adding_mask(blurred_frame, tuple(config['red_lower']), tuple(config['red_upper']))
        frame_contours = fr.find_contours(masked_frame.copy())
        frame = fr.draw_circle(frame_contours, frame)
        key = cv.waitKey(20) & 0xFF
        if key == 27:
            break
        cv.imshow("Detect_Circle", frame)
        cv.imshow("masked_frame", masked_frame)
    return vid_src


if __name__ == "__main__":
    video_source = main()
    video_source.release()
    # close all windows
    cv.destroyAllWindows()

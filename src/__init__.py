from src.config_parser import load_config
from frame_methods import modified_frame as fm
import cv2 as cv
import os


config = load_config("config")
cwd = os.getcwd().rstrip('\src')
cwd = cwd + r"\data\videoplayback.mp4"


def main():
    vid_src = cv.VideoCapture(cwd)
    while True:
        ret, frame = vid_src.read()
        if frame is None:
            break
        blur_added_frame = fm.add_artifacts(frame)
        masked_frame = fm.create_mask_to_video_frame(blur_added_frame, tuple(config['red_lower']), tuple(config['red_upper']))
        contours = fm.find_contours(masked_frame)
        add_circle = fm.draw_circle(contours, vid_src)
        cv.imshow("Detect_Circle", add_circle )
        key = cv.waitKey(10) & 0xFF
        if key == 27:
            break
    return vid_src


if __name__ == "__main__":
    vs = main()
    vs.release()
# close all windows
    cv.destroyAllWindows()
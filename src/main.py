from config_parser import load_config
from frame_methods import frame_revamp as fr
import cv2 as cv
import imutils
import time

config = load_config("config")
# cwd = os.getcwd().rstrip('\src')
# cwd = cwd + r"\data\videoplayback.mp4"


def main():
    vid_src = cv.VideoCapture(config['path_to_file'])
    time.sleep(2.0)
    while True:
        ret, frame = vid_src.read()
        if not ret or frame is None:
            break
        frame = imutils.resize(frame, width=600)
        blurred_frame = fr.adding_artifact(frame)
        masked_frame = fr.adding_mask(blurred_frame, tuple(config['red_lower']), tuple(config['red_upper']))
        frame_contours = fr.find_contours(masked_frame)
        frame = fr.draw_circle(frame_contours, frame)
        key = cv.waitKey(10) & 0xFF
        if key == 27:
            break
        cv.imshow("Detect_Circle", frame)
        # cv.imshow("masked_frame", masked_frame)
    vid_src.release()
    return None


if __name__ == "__main__":
    main()
    # close all windows
    cv.destroyAllWindows()

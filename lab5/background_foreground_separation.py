import cv2
import cv2.cv as cv
import numpy as np


def get_background(vid, fr_count):
    _, img = vid.read()
    avg_img = np.float32(img)
    for fr in range(1, fr_count):
        _, img = vid.read()
        alpha = 1.0 / (fr + 1)
        cv2.accumulateWeighted(img, avg_img, alpha)

        norm_img = cv2.convertScaleAbs(avg_img)
        cv2.imshow('img', img)
        cv2.imshow('normImg', norm_img)
        print "fr = ", fr, " alpha = ", alpha

    cv2.imwrite('background.jpg', norm_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    vid.release()

cap = cv2.VideoCapture('traffic.mp4')
frame_width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
frame_per_second = int(cap.get(cv.CV_CAP_PROP_FPS))
frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
print frame_width, frame_height, frame_per_second, frame_count
get_background(cap, frame_count)

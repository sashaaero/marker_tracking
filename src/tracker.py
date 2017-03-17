import cv2
import numpy as np

from asift.asift import affine_detect, Detection
from asift.find_obj import init_feature, filter_matches, explore_match
from asift.common import Timer
from collections import namedtuple
from itertools import product
from multiprocessing.pool import ThreadPool


def keypressed(key):
    return cv2.waitKey(1) == key


def init_detection(original, detector):
    pass


def track_diff(state, img, detector, matcher):
    good = False

    return good, None


def draw_detection(detection):
    pass



def track(cam, original):
    ret_val, img = cam.read()
    prev_detection = init_detection(original, img)

    while True:
        ret_val, img = cam.read()

        good, detection = track_diff(prev_detection, img)

        if not good:
            detection = init_detection(original, img)

        draw_detection(detection, img)

        if keypressed(27):
            break

        prev_state = detection


def get_original(cam):
    while True:
        _, img = cam.read()

        if keypressed(13):
            return img
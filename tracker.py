from collections import namedtuple

import cv2


def keypressed(key):
    return cv2.waitKey(1) == key


def init_state(original):
    pass


def track_diff(state, img):
    pass


def draw_state(state):
    pass


State = namedtuple("State", [])


def track(cam, original):
    ret_val, img = cam.read()
    prev_state = init_state(original, img)

    while True:
        ret_val, img = cam.read()

        good, state = track_diff(prev_state, img)

        if not good:
            print("Calculating new state")
            state = init_state(original, img)


        draw_state(state, img)

        if keypressed(27):
            break

        prev_state = state


def get_original(cam):
    while True:
        _, img = cam.read()

        if keypressed(13):
            return img
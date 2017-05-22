import cv2
import numpy as np


COLOR_WHITE = (255, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)


def capture_img(cam, size=(640, 480)):
    _, img = cam.read()
    return cv2.resize(img, size)


def capture_orig(cam):
    while True:
        img = capture_img(cam)

        cv2.imshow("Press enter to capture", cv2.flip(img, 1))
        if cv2.waitKey(1) == 13:
            return img


def key_pressed(key=None, wait=1):
    pressed = cv2.waitKey(wait)
    if key is None:
        return 0 <= pressed < 255
    else:
        return pressed == key


def wait_for_key(key=None):
    while not key_pressed(key):
        pass


def draw_poly(img, corners, color=COLOR_WHITE):
    corners = [np.int32(corners)]
    cv2.polylines(img, corners, True, color)

from collections import namedtuple

import cv2
import numpy as np
from asift.asift import affine_detect, Detection
from asift.find_obj import init_feature, filter_matches
from asift.common import Timer
from itertools import product
from multiprocessing.pool import ThreadPool


def mix(src1, src2, k=0.5):
    return src1 * k + src2 * (1 - k)


def draw_match_bounds(original_shape, img, H):
    def line(p1, p2, color):
        cv2.line(img, tuple(p1), tuple(p2), color)

    h, w = original_shape[:2]
    if H is not None:
        corners = np.float32([[1, 1], [w, 1], [w, h], [1, h]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))
        line(corners[0], corners[1], (0, 0, 255))  # red top
        line(corners[1], corners[2], (0, 255, 0))
        line(corners[2], corners[3], (0, 255, 0))
        line(corners[3], corners[0], (0, 255, 0))


def capture_img(cam):
    while True:
        _, img = cam.read()
        cv2.imshow("Press enter to capture", img)
        if cv2.waitKey(1) == 13:
            return img

Match = namedtuple("match", ["match", "parameters"])


def detect_copy(cam, orig):
    pool = ThreadPool(processes=cv2.getNumberOfCPUs())

    detector, matcher = init_feature("surf-flann")

    # orig = cv2.imread("img/card.jpg")
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    orig = cv2.resize(orig, (0, 0), fx=0.5, fy=0.5)
    # orig = cv2.imread("img/triangles.png")
    # orig = cv2.imread("img/orig.jpg")
    detection1 = affine_detect(detector, orig, pool=pool)

    previous_match = Match(1, (0, 0))

    while True:
        ret_val, img0 = cam.read()

        img = cv2.resize(img0, (0, 0), fx=0.5, fy=0.5)

        dev = img
        dev = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # dev = cv2.adaptiveThreshold(dev, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        #
        # dev = cv2.morphologyEx(dev, cv2.MORPH_CLOSE, kernel)

        with Timer("detecting"):
            detection2 = affine_detect(detector, dev, pool=pool)
            # print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))

        with Timer('matching'):
            detection2.sort(key=lambda d: (d.parameters[0] - previous_match.parameters[0],
                                           d.parameters[1] - previous_match.parameters[1]))

            best_match = -1
            best_H = None
            best_params = (0, 0)
            for i, (d1, d2) in enumerate(product(detection1, detection2)):
                raw_matches = matcher.knnMatch(d1.descriptors, trainDescriptors=d2.descriptors, k=2)  # 2

                p1, p2, kp_pairs = filter_matches(d1.key_points, d2.key_points, raw_matches, ratio=0.5)

                match = len(kp_pairs) * 100.0 / len(raw_matches)

                # print("matched {}%".format(match))

                if len(p1) >= 4:
                    H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                    if match > best_match:
                        best_H = H
                        best_match = match
                        best_params = d2.parameters

                        if match / previous_match.match >= 0.95:
                            print("stopped at i={}".format(i))
                            break

            previous_match = Match(best_match, best_params)

            if best_H is not None:
                draw_match_bounds(np.shape(orig), img, best_H)



        # cv2.imshow('my webcam', img0)
        cv2.imshow('my dev', dev)
        cv2.imshow('my img', img)

        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()


def main():
    cam = cv2.VideoCapture(1)

    orig = capture_img(cam)
    detect_copy(cam, orig)

if __name__ == '__main__':
    main()
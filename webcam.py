from collections import namedtuple

import cv2
import numpy as np
from asift.asift import affine_detect, Detection
from asift.find_obj import init_feature, filter_matches, explore_match
from asift.common import Timer
from itertools import product
from multiprocessing.pool import ThreadPool


def mix(src1, src2, k=0.5):
    return src1 * k + src2 * (1 - k)


def draw_points(img, points, color=(0, 255, 255)):
    for p in points:
        x = int(p[0])
        y = int(p[1])
        cv2.circle(img, (x, y), 2, color)
        cv2.circle(img, (x, y), 3, color)
        cv2.circle(img, (x, y), 4, color)


def draw_match_bounds(original_shape, img, H):
    def line(p1, p2, color):
        cv2.line(img, tuple(p1), tuple(p2), color)

    h, w = original_shape[:2]

    if H is not None:
        corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))
        line(corners[0], corners[1], (0, 0, 255))  # red top
        line(corners[1], corners[2], (0, 255, 0))
        line(corners[2], corners[3], (0, 255, 0))
        line(corners[3], corners[0], (0, 255, 0))


def capture_img(cam, size=(640, 480)):
    _, img = cam.read()
    return cv2.resize(img, size)


def capture_orig(cam):
    while True:
        img = capture_img(cam)

        cv2.imshow("Press enter to capture", cv2.flip(img, 1))
        if cv2.waitKey(1) == 13:
            return img


def key_pressed(key):
    return cv2.waitKey(1) == key


def wait_for_key(key):
    while not key_pressed(key):
        pass

Match = namedtuple("match", ["match", "parameters"])


# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def detect_copy_klt(cam, orig):
    pool = ThreadPool(processes=cv2.getNumberOfCPUs())

    def refresh_keypoints(detection1, frame):
        detection2 = affine_detect(detector, frame, pool=pool)

        best = 0
        rslt = None, None
        for i, (d1, d2) in enumerate(product(detection1, detection2)):
            raw_matches = matcher.knnMatch(d1.descriptors, trainDescriptors=d2.descriptors, k=2)  # 2

            p1, p2, kp_pairs = filter_matches(d1.key_points, d2.key_points, raw_matches, ratio=0.75)

            if len(kp_pairs) >= best:
                rslt = p1, p2
                best = len(kp_pairs)

        return rslt

    def extract_points(kp_pairs, idx):
        return list(p[idx].pt for p in kp_pairs)

    def mean(pts):
        m = (0, 0)
        for (x, y) in pts:
            m = m[0] + x, m[1] + y

        return m[0] / len(pts), m[1] / len(pts)

    # ret_val, img0 = cam.read()
    img0 = orig.copy()

    old_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    detector, matcher = init_feature("sift")
    detection0 = affine_detect(detector, old_gray)

    orig_points = list(map(
        lambda kp: (kp.pt[0], kp.pt[1]),
        detection0[0].key_points
    ))

    draw_points(img0, orig_points)
    cv2.imshow('original', img0)

    old_points = None
    frames_till_refresh = 0
    while True:
        img0 = capture_img(cam)

        frame_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

        if frames_till_refresh <= 0:
            print("Refreshing")
            old_points, new_points = refresh_keypoints(detection0, frame_gray)

            img1 = img0.copy()
            draw_points(img1, new_points)
            cv2.imshow("update", img1)

            # wait_for_key(13)

            frames_till_refresh = 60
        else:
            new_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **lk_params)

            if new_points is not None:
                # new_points = new_points[st == 1]
                pass
            else:
                new_points = []
                print("No good features found")


        if new_points is not None and len(new_points) > 0:
            H, status = cv2.findHomography(old_points, new_points, cv2.RANSAC, 5.0)

            new_points = np.array([p for i, p in enumerate(new_points) if status[i]])

            draw_match_bounds(orig.shape, img0, H)

            draw_points(img0, new_points, color=(255, 0, 255))
            draw_points(img0, [mean(new_points)], color=(0, 255, 0))

        cv2.imshow('my img', cv2.flip(img0, 1))

        if key_pressed(27):
            break  # esc to quit

        old_gray = frame_gray.copy()
        old_points = new_points

        frames_till_refresh -= 1

    cv2.destroyAllWindows()


def main():
    cv2.ocl.setUseOpenCL(True)


    cam = cv2.VideoCapture(0)

    orig = capture_orig(cam)
    detect_copy_klt(cam, orig)

    cam.release()

if __name__ == '__main__':
    main()

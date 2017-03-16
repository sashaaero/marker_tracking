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


def draw_match_bounds(original_shape, img, H, color_top=(0, 0, 255), color_other=(0, 255, 0)):
    def line(p1, p2, color):
        cv2.line(img, tuple(p1), tuple(p2), color)

    h, w = original_shape[:2]

    if H is not None:
        corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))
        line(corners[0], corners[1], color_top)  # red top
        line(corners[1], corners[2], color_other)
        line(corners[2], corners[3], color_other)
        line(corners[3], corners[0], color_other)


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
lk_params = dict(winSize=(15, 10),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))

REFRESH_INTERVAL = 90


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

    orig_points = np.array(list(map(
        lambda kp: np.array([[np.float32(kp.pt[0]), np.float32(kp.pt[1])]]),
        detection0[0].key_points
    )))

    # draw_points(img0, orig_points[:, 0])
    # cv2.imshow('original', img0)

    old_points = None
    best_points = None
    bad_points = None
    frames_till_refresh = 0
    H = None
    H2 = None
    while True:
        img0 = capture_img(cam)

        frame_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

        # hsv = np.zeros_like(img0)
        # hsv[..., 1] = 255
        #
        # with Timer("flow"):
        #     flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, .5, 1, 3, 5, 5, 1.1, 0)
        #
        # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv[...,0] = ang * 180 / np.pi / 2
        # hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # cv2.imshow('frame2', bgr)
        # wait_for_key(13)

        if frames_till_refresh <= 0:
            with Timer("Refreshing"):
                old_points, new_points = refresh_keypoints(detection0, frame_gray)

            refresh_points = old_points.copy()

            img1 = img0.copy()
            draw_points(img1, new_points)
            cv2.imshow("update", img1)

            H, status = cv2.findHomography(old_points, new_points, cv2.RANSAC, 5.0)

            H2 = H

            # wait_for_key(13)
            if H is not None:
                frames_till_refresh = REFRESH_INTERVAL
            else:
                print("can't find homography")
        else:
            new_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **lk_params)

            H1, status = cv2.findHomography(refresh_points, new_points, cv2.RANSAC, 5.0)

            if status is not None:
                # print(status)
                best_points = np.array([p for i, p in enumerate(new_points) if status[i] > 0])
                bad_points = np.array([p for i, p in enumerate(new_points) if status[i] == 0])
                # print("{}={}/{}".format(sum(status), len(best_points), len(status)))
                # postpone refreshing by one frame
                if len(best_points) * 2 > len(new_points):
                    frames_till_refresh += 1
                    print("+1")
            else:
                best_points = []
            if H1 is not None:
                H = H1 #H1.dot(H)
                new_points = cv2.perspectiveTransform(refresh_points.reshape(1, -1, 2), H).reshape(-1, 2)

            H3, status = cv2.findHomography(old_points, new_points, cv2.RANSAC, 5.0)
            if H3 is not None:
                H2 = H3.dot(H2)

            if new_points is not None:
                # new_points = new_points[st == 1]
                pass
            else:
                new_points = []
                print("No good features found")
                frames_till_refresh = 0

        if new_points is None or len(new_points) == 0:
            new_points = [[]]


        # else:
        #     new_points = np.array([p for i, p in enumerate(new_points) if status[i]])

        draw_match_bounds(orig.shape, img0, H)

        # cumulative homography matrix
        draw_match_bounds(orig.shape, img0, H2, color_top=(0, 255, 255), color_other=(255, 0, 255))

        if len(new_points) > 0:
            draw_points(img0, new_points, color=(255, 0, 255))
            if best_points is not None:
                draw_points(img0, best_points, color=(255, 255, 255))

            if bad_points is not None:
                draw_points(img0, bad_points, color=(0, 0, 255))

            draw_points(img0, [mean(new_points)], color=(0, 255, 0))

        cv2.imshow('my img', cv2.flip(img0, 1))

        # wait_for_key(ord('z'))

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

    # cam = cv2.VideoCapture("test.avi")

    orig = cv2.imread("sample_gaga.jpg")

    detect_copy_klt(cam, orig)

    cam.release()

if __name__ == '__main__':
    main()

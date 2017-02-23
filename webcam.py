import cv2
import numpy as np
from asift.asift import affine_detect
from asift.find_obj import init_feature, filter_matches
from asift.common import Timer
from multiprocessing.pool import ThreadPool

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))


def mix(src1, src2, k=0.5):
    return src1 * k + src2 * (1 - k)


def draw_match_bounds(original_shape, img, H):
    def line(p1, p2, color):
        cv2.line(img, tuple(p1), tuple(p2), color)

    h, w = original_shape[:2]
    if H is not None:
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        # print(corners.reshape(1, -1, 2), H)
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


def detect_copy(cam, orig):
    pool = ThreadPool(processes=cv2.getNumberOfCPUs())

    detector, matcher = init_feature("surf-flann")

    #with Timer():
    # orig = cv2.imread("img/card.jpg")
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    orig = cv2.resize(orig, (0, 0), fx=0.5, fy=0.5)
    # orig = cv2.imread("img/triangles.png")
    # orig = cv2.imread("img/orig.jpg")
    kp1, desc1 = affine_detect(detector, orig, pool=pool)

    while True:
        ret_val, img0 = cam.read()

        img = cv2.resize(img0, (0, 0), fx=0.5, fy=0.5)

        dev = img
        dev = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # dev = cv2.adaptiveThreshold(dev, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        #
        # dev = cv2.morphologyEx(dev, cv2.MORPH_CLOSE, kernel)

        with Timer("detecting"):
            kp2, desc2 = affine_detect(detector, dev, pool=pool)
            # print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))

        with Timer('matching'):
            if len(desc2) > 0:
                raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2

                p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches, ratio=0.5)

                print("matched: {}/{}".format(len(kp_pairs), len(raw_matches)))

                if len(p1) >= 4:
                    H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)

                    # print(status)

                    draw_match_bounds(np.shape(orig), img, H)
            else:
                print("COME ON!!!!")


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
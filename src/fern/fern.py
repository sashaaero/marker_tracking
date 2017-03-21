import random
from collections import defaultdict

import cv2
import numpy as np

from asift.asift import affine_skew, get_camera_params
from webcam import wait_for_key

#
# Article: OZUYSAL ET AL.: FAST KEYPOINT RECOGNITION USING RANDOM FERNS
# Link: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4760148
#


class Fern:
    def __init__(self, size, key_point_pairs, feature_function):
        self._size = size
        self._kp_pairs = key_point_pairs
        self._function = feature_function

    def calculate(self, sample):
        """ Calculates feature function on all kp_pairs and generates binary number according to article"""
        result = 0
        for p1, p2 in self._kp_pairs:
            if self._function(p1, p2, sample):
                result += 1

            result *= 2

        return result


class FernDetector:
    def __init__(self, img, patch_size=(32, 32)):
        self._patch_size = patch_size
        self._init_ferns()
        self._train(img)

    @staticmethod
    def _calc_feature(kp1, kp2, img):
        return 1 if img[int(kp1[1]), int(kp1[0])] < img[int(kp2[1]), int(kp2[0])] else 0

    def _init_ferns(self, fern_est_size=10):
        kp_pairs = list(self._generate_key_point_pairs())
        n = len(kp_pairs)

        # maps key_point[i] to fern[fern_indices[i]]
        fern_indices = []

        # generate n // fern_est_size Ferns
        num_ferns = n // fern_est_size
        for fern_index in range(num_ferns):
            fern_indices += [fern_index] * fern_est_size

        # increase last fern size if needed
        if len(fern_indices) < n:
            fern_indices += [num_ferns - 1] * (n - len(fern_indices))

        random.shuffle(fern_indices)

        fern_kp_pairs = defaultdict(list)
        for kp_idx, fern_idx in enumerate(fern_indices):
            fern_kp_pairs[fern_idx].append(kp_pairs[kp_idx])

        self._ferns = [Fern(self._patch_size, kp_pairs, self._calc_feature) for fern_idx, kp_pairs in fern_kp_pairs.items()]

    def _train(self, train_img):
        img_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

        corners = cv2.goodFeaturesToTrack(img_gray, maxCorners=500, qualityLevel=0.01, minDistance=16)[:15]
        self._classes_count = len(corners)

        K = 2**11
        self._fern_p = np.zeros((len(self._ferns), K, self._classes_count))
        self.key_points = []

        for class_idx, (corner, ) in enumerate(corners):
            print("Training class {} / {}".format(class_idx + 1, self._classes_count))
            self.key_points.append(corner)
            N = [0] * K
            for fern_idx, fern in enumerate(self._ferns):
                for patch in self._generate_patch_class(img_gray, corner):
                    k = fern.calculate(patch)
                    assert k < K, "WTF!!!"
                    N[k] += 1

                self._fern_p[fern_idx, k, class_idx] = N[k]

        for fern_idx in range(len(self._ferns)):
            for cls_idx in range(self._classes_count):
                Nc = np.sum(self._fern_p[fern_idx, :, cls_idx])
                for k in range(K):
                    self._fern_p[fern_idx, k, cls_idx] = (self._fern_p[fern_idx, k, cls_idx] + 1) / (Nc + K)

    def match(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(image, maxCorners=500, qualityLevel=0.01, minDistance=16)

        key_points_trained = []
        key_points_matched = []
        key_points_pairs = []

        for (corner, ) in corners:
            probs = np.ones((self._classes_count, ))

            patch = self._generate_patch(image, corner)
            for fern_idx, fern in enumerate(self._ferns):
                k = fern.calculate(patch)
                for class_idx in range(self._classes_count):
                    probs[class_idx] *= self._fern_p[fern_idx, k, class_idx]

            most_probable_class = 0
            most_prob = probs[most_probable_class]
            for class_idx in range(1, self._classes_count):
                if most_prob < probs[class_idx]:
                    most_probable_class = class_idx
                    most_prob = probs[most_probable_class]


            key_points_trained.append(self.key_points[most_probable_class])
            key_points_matched.append(corner)
            key_points_pairs.append((self.key_points[most_probable_class], corner))

        return key_points_trained, key_points_matched, key_points_pairs

    def _generate_key_point_pairs(self, small_k=0.3, large_k=0.7):
        pw, ph = self._patch_size

        rx_small = pw * small_k / 2
        ry_small = ph * small_k / 2

        rx_large = pw * large_k / 2
        ry_large = ph * large_k / 2

        for angle in range(0, 180, 2):
            phi  = angle / 180 * np.math.pi
            phi1 = (angle + 180) / 180 * np.math.pi

            yield (rx_small * np.math.cos(phi),  ry_small * np.math.sin(phi)), \
                  (rx_small * np.math.cos(phi1), ry_small * np.math.sin(phi1))

            yield (rx_large * np.math.cos(phi),  ry_large * np.math.sin(phi)), \
                  (rx_large * np.math.cos(phi1), ry_large * np.math.sin(phi1))

    def _generate_patch(self, img, corner):
        x, y = corner
        h, w = np.shape(img)

        pw, ph = self._patch_size
        pw2 = pw // 2 + 1
        ph2 = ph // 2 + 1

        # ensure patch is inside image
        cx = min(max(int(x), pw2), w - 1 - pw2)
        cy = min(max(int(y), ph2), h - 1 - ph2)
        # top left corner
        x0, y0 = cx - pw2, cy - ph2

        return img[y0:y0 + ph, x0:x0 + pw]

    def _generate_patch_class(self, img, corner):
        """ generate patch transformations """
        patch = self._generate_patch(img, corner)

        params = get_camera_params()

        for (t, phi) in params:
            patch, mask, Ai = affine_skew(t, phi, img)
            yield patch


class FernMatcher:
    pass



def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0]))
        p2.append(np.int32(np.array(kpp[1]) + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(win, vis)

    return vis



if __name__ == "__main__":
    orig = cv2.imread("../../sample_gaga.jpg")
    orig2 = cv2.flip(orig, 1)

    detector = FernDetector(orig)
    kp1, kp2, kp_p = detector.match(orig2)

    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    orig2 = cv2.cvtColor(orig2, cv2.COLOR_BGR2GRAY)

    explore_match("qqq", orig, orig2, kp_p)

    wait_for_key(ord('q'))

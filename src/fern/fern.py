import random
from collections import defaultdict

import cv2
import numpy as np

from asift.asift import affine_skew, get_camera_params

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
    def _calc_feature(self, kp1, kp2, img):
        return 1 if img[kp1] < img[kp2] else 0

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

        self._ferns = [Fern(self._patch_size, kp_pairs, self._calc_feature) for fern_idx, kp_pairs in fern_kp_pairs]

    def _train(self, train_img):
        img_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

        corners = cv2.goodFeaturesToTrack(img_gray, maxCorners=500, qualityLevel=0.01, minDistance=16)
        self._classes_count = len(corners)

        K = 2**11
        self._fern_p = np.zeros((len(self._ferns), K, self._classes_count))

        for class_idx, (corner, ) in enumerate(corners):
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

    def _match(self, image):
        corners = cv2.goodFeaturesToTrack(image, maxCorners=500, qualityLevel=0.01, minDistance=16)

        for (corner, ) in corners:
            probs = np.ones((self._classes_count, ))

            patch = self._generate_patch(image, corner)
            for fern_idx, fern in enumerate(self._ferns):
                k = fern.calculate(patch)
                for class_idx in range(self._classes_count):
                    probs[class_idx] *= self._fern_p[fern_idx, k, class_idx]

            # TODO: find argmax -> most_probable class_idx, then what?



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
        cx = min(max(x, pw2), w - 1 - pw2)
        cy = min(max(y, ph2), h - 1 - ph2)
        # top left corner
        x0, y0 = cx - pw2, cy - ph2

        return img[y0:y0 + ph, x0:x0 + pw]

    def _generate_patch_class(self, img, corner):
        """ generate patch transformations """
        patch = self._generate_patch(img, corner)

        params = get_camera_params()

        for (t, phi) in params:
            yield affine_skew(t, phi, patch)


class FernMatcher:
    pass


if __name__ == "__main__":
    orig = cv2.imread("../sample.jpg")
    FernDetector(orig)

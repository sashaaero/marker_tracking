import random

import cv2
import numpy as np

from asift.asift import affine_skew, get_camera_params


class FernDetector:
    def __init__(self, img, patch_size=(32, 32)):
        self._patch_size = patch_size
        self._train(img)

    def _train(self, train_img):
        img_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

        corners = cv2.goodFeaturesToTrack(img_gray, maxCorners=500, qualityLevel=0.01, minDistance=16)

        self._kp_pairs = list(self._generate_key_point_pairs())
        self._ferns = self._generate_fern_indices(self._kp_pairs)

        classes = []
        for (corner, ) in corners:
            for img in self._generate_patch_class(corner, img_gray):
                for idx, (kp1, kp2) in enumerate(self._kp_pairs):
                    fern = self._ferns[idx]

    def _calc_feature(self, img, kp1, kp2):


    def _generate_fern_indices(self, key_points, fern_est_size=10):
        n = len(key_points)
        indices = []
        fern_index = 0
        for _ in range(fern_est_size):
            indices += [fern_index] * (n // fern_est_size)
            fern_index += 1

        fern_index -= 1

        if len(indices) < n:
            indices += [fern_index] * (n - len(indices))

        random.shuffle(indices)

        {}

        return indices

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

    def _generate_patch_class(self, corner, img):
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

        patch = img[y0:y0 + ph, x0:x0 + pw]

        params = get_camera_params()

        for (t, phi) in params:
            yield affine_skew(t, phi, patch)










class FernMatcher:
    pass


if __name__ == "__main__":
    orig = cv2.imread("../sample.jpg")
    FernDetector(orig)

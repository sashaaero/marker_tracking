from typing import IO

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import util

from collections import defaultdict


class Fern:
    def __init__(self, size, key_point_pairs):
        self._size = size
        self.kp_pairs = key_point_pairs

    def calculate(self, sample):
        """ Calculates feature function on all kp_pairs and generates binary number according to article"""
        result = 0

        for (y1, x1), (y2, x2) in self.kp_pairs:
            result = (result + (0 if (sample[y1, x1] < sample[y2, x2]) else 1)) * 2

        return result

    def draw(self, k, sample):
        levels = []

        for _ in range(len(self.kp_pairs)):
            levels.append(k % 2)
            k >>= 1

        levels.reverse()
        for ((y1, x1), (y2, x2)), level in zip(self.kp_pairs, levels):
            sample[y1, x1] = 255 * level
            sample[y2, x2] = 255 * (1 - level)


class FernDetector:
    def __init__(self, sample, patch_size=(16, 16), max_train_corners=40, max_match_corners=200):
        self._patch_size = patch_size
        self._max_train_corners = max_train_corners
        self._max_match_corners = max_match_corners
        self._init_ferns()
        self._train(sample)

    def _init_ferns(self, fern_est_size=11):
        kp_pairs = list(util.generate_key_point_pairs(self._patch_size))
        n = len(kp_pairs)

        # maps key_point[i] to fern[fern_indices[i]]
        fern_indices = []

        # generate n // fern_est_size Ferns
        num_ferns = n // fern_est_size
        for fern_index in range(num_ferns):
            fern_indices += [fern_index] * fern_est_size

        self._S = fern_est_size
        # increase last fern size if needed
        if len(fern_indices) < n:
            self._S += n - len(fern_indices)
            fern_indices += [num_ferns - 1] * (n - len(fern_indices))

        random.shuffle(fern_indices)

        fern_kp_pairs = defaultdict(list)
        for kp_idx, fern_idx in enumerate(fern_indices):
            fern_kp_pairs[fern_idx].append(kp_pairs[kp_idx])

        self._ferns = [Fern(self._patch_size, kp_pairs) for fern_idx, kp_pairs in fern_kp_pairs.items()]

    def _train(self, train_img):
        img_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
        H, W = np.shape(img_gray)[:2]

        corners = list(util.get_stable_corners(img_gray, self._max_train_corners))

        img_gray = cv2.GaussianBlur(img_gray, (7, 7), 25)

        img1 = img_gray.copy()
        for y, x in corners:
            cv2.circle(img1, (x, y), 3, util.COLOR_WHITE, -1)

        cv2.imshow("corners", img1)
        util.wait_for_key()

        self._classes_count = len(corners)

        K = 2 ** (self._S + 1)
        self._fern_p = np.zeros((len(self._ferns), self._classes_count, K))
        self.key_points = []

        title = "Training {} classes".format(self._classes_count)
        for class_idx, corner in enumerate(util.iter_timer(corners, title=title, print_iterations=True)):
            self.key_points.append(corner)

            cy, cx = corner
            assert 0 <= cy <= H and 0 <= cx <= W, "(cy, cx)=({}, {}) (H, W)=({}, {})".format(cy, cx, H, W)

            patch_class = list(util.generate_patch_class(img_gray, corner, self._patch_size))
            self._draw_patch_class(patch_class, class_idx)

            for patch in patch_class:
                for fern_idx, fern in enumerate(self._ferns):
                    k = fern.calculate(patch)
                    assert 0 <= k < K, "WTF!!!"
                    self._fern_p[fern_idx, class_idx, k] += 1

        Nr = 1

        for fern_idx in util.iter_timer(range(len(self._ferns)), title="Calculating probs"):
            for cls_idx in range(self._classes_count):
                Nc = np.sum(self._fern_p[fern_idx, cls_idx, :])
                self._fern_p[fern_idx, cls_idx, :] += Nr
                self._fern_p[fern_idx, cls_idx, :] /= (Nc + K * Nr)

                print("P_min={}, P_max={}"
                      .format(np.min(self._fern_p[fern_idx, cls_idx, :]),
                              np.max(self._fern_p[fern_idx, cls_idx, :]))
                      )

        self._fern_p = np.log(self._fern_p)

        print("Training complete!")

    def match(self, image):
        dims = len(np.shape(image))
        if dims == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        with util.Timer("track features"):
            corners = util.get_corners(image, self._max_match_corners)

        image = cv2.GaussianBlur(image, (7, 7), 25)

        key_points_trained = []
        key_points_matched = []
        key_points_pairs = []

        for corner in util.iter_timer(corners, title="Matching corners", print_iterations=False):
            probs = np.zeros((self._classes_count,))

            patch = util.generate_patch(image, corner, self._patch_size)
            for fern_idx, fern in enumerate(self._ferns):
                k = fern.calculate(patch)
                probs += self._fern_p[fern_idx, :, k]

            most_probable_class = np.argmax(probs)
            best_key_point = self.key_points[most_probable_class]

            key_points_trained.append(best_key_point)
            key_points_matched.append(corner)
            key_points_pairs.append((best_key_point, corner))

        return util.flip_points(key_points_trained), \
               util.flip_points(key_points_matched), \
               key_points_pairs

    def detect(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kp_t, kp_m, kpp = self.match(image)
        H, status = cv2.findHomography(kp_t, kp_m, cv2.RANSAC, 5.0)

        h, w = np.shape(image)
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        if H is not None:
            return util.transform32(corners, H), H

        return [], H

    def _draw_patch_class(self, patches, cls_idx):
        w, h = self._patch_size

        W = (w + 5) * 10
        H = (h + 5) * (len(patches) // 10 + 1)

        img = np.zeros((W, H))
        for idx, patch in enumerate(patches):
            x = (idx // 10) * (w + 5)
            y = (idx % 10) * (h + 5)

            img[y:y + h, x: x + w] = patch

        cv2.imwrite("img/train/cls{}.png".format(cls_idx), img)

    def draw_learned_ferns(self):
        w, h = self._patch_size
        ferns_count = len(self._ferns)

        _, K, _ = self._fern_p.shape

        mask = np.zeros((h, w), np.uint8)
        for fern in self._ferns:
            for (y1, x1), (y2, x2) in fern.kp_pairs:
                mask[y1, x1] = 255
                mask[y2, x2] = 255

        mask = 255 - mask

        for cls_idx in util.iter_timer(range(self._classes_count), title="Drawing ferns"):
            img = np.zeros(((1 + (ferns_count // 10)) * (h + 5), (w + 5) * 10)) * 128

            for fern_idx, fern in enumerate(self._ferns):
                x0 = (fern_idx % 10) * (w + 5)
                y0 = (fern_idx // 10) * (h + 5)

                k = np.argmax(self._fern_p[fern_idx, :, cls_idx])

                sample = np.ones((h, w), np.uint8) * 128
                fern.draw(k, sample)

                # mask = sample.copy()
                # # invert
                # mask = mask + 128
                # _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)

                img[y0:y0 + h, x0: x0 + w] = sample  # cv2.inpaint(sample, mask, 2, cv2.INPAINT_TELEA)

                # for k in range(K):
                #     p = self._fern_p[fern_idx, k, cls_idx]

            cv2.imwrite("img/learn/cls{}.png".format(cls_idx), img)

    def draw_learned_ferns_2(self, path=""):
        print("Drawing grphics..")
        ferns_count, K, _ = self._fern_p.shape
        x = list(range(K))

        for cls_idx in range(self._classes_count):
            f, axes_arr = plt.subplots(ferns_count,
                                       figsize=(6, 1.5 * ferns_count),
                                       sharey=True,
                                       sharex=True)

            for fern_idx, fern in enumerate(self._ferns):
                axes_arr[fern_idx].plot(x, self._fern_p[fern_idx, :, cls_idx])

            # f.subplots_adjust(hspace=0)

            plt.savefig("{}plot_cls{}.png".format(path, cls_idx), dpi=100)
            plt.close(f)
        print("All graphs were drawn")

    def serialize(self, file: IO):
        file.write(str(len(self._ferns)))
        lines = [
            " ".join((" ".join(y2, x2, y1, x1) for (y2, x2), (y1, x1) in fern.kp_pairs))
            for fern in self._ferns
        ]
        file.writelines(lines)



import random
from collections import defaultdict, namedtuple

import cv2
import numpy as np

import util
from asift.common import Timer, iter_timer
import matplotlib.pyplot as plt

from util import wait_for_key, key_pressed, explore_match, mult, COLOR_WHITE
from webcam import draw_match_bounds

Z = 40


class Collector:
    def __init__(self, point):
        self._point_sum = np.array(point)
        self._count = 1

    def push(self, new_point):
        new_point = np.array(new_point)
        assert new_point.shape[0] == 2

        self._point_sum += new_point
        self._count += 1

    def dist2(self, point):
        point = np.array(point)
        assert point.shape[0] == 2

        return np.sum((self._point_sum - point) ** 2)

    @property
    def x(self):
        return self._point_sum[1] / self._count

    @property
    def y(self):
        return self._point_sum[0] / self._count

    @property
    def count(self):
        return self._count

    def __str__(self):
        return "Collector(x:{}, y:{}, cnt:{})".format(self.x, self.y, self._count)


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
        kp_pairs = list(self._generate_key_point_pairs())
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

    def _get_stable_corners(self, train_img, max_corners=100):
        for c in self._get_corners(train_img, max_corners):
            yield np.int32(c)

        return

        H, W = np.shape(train_img)[:2]

        def find_collector(collectors, point, radius=2):
            radius2 = radius ** 2

            min_c = None
            for c in collectors:
                dist = c.dist2(point)
                if dist < radius2:
                    min_c = c
                    radius2 = dist

            return min_c

        collectors = []
        for R_inv, img in FernDetector._generate_affine_deformations(train_img, theta_step=36, deformations=3):
            corners = np.array([list(self._get_corners(img, 500))])

            (corners_inv, ) = cv2.transform(corners, R_inv)

            for corner in corners_inv:
                corner[1], corner[0] = corner[0], corner[1]
                cy, cx = corner

                if not (0 <= cy < H and 0 <= cx < W):
                    continue

                # assert 0 <= cy < H and 0 <= cx < W, "(cy, cx)=({}, {}) (H, W)=({}, {})".format(cy, cx, H, W)

                collector = find_collector(collectors, corner)

                if collector is None:
                    collectors.append(Collector(corner))
                else:
                    collector.push(corner)

        collectors = sorted(collectors, key=lambda c: -c.count)

        for c in collectors[:max_corners]:
            yield (int(c.y), int(c.x))

    def _train(self, train_img):
        img_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
        H, W = np.shape(img_gray)[:2]

        corners = list(self._get_stable_corners(img_gray, self._max_train_corners))

        img1 = img_gray.copy()
        for corner in corners:
            y, x = corner
            cv2.circle(img1, (x, y), 3, COLOR_WHITE, -1)

        cv2.imshow("corners", img1)

        self._classes_count = len(corners)

        K = 2 ** (self._S + 1)
        self._fern_p = np.zeros((len(self._ferns), self._classes_count, K))
        self.key_points = []

        title = "Training {} classes".format(self._classes_count)
        for class_idx, corner in enumerate(iter_timer(corners, title=title, print_iterations=True)):
            self.key_points.append(corner)

            cy, cx = corner
            assert 0 <= cy <= H and 0 <= cx <= W, "(cy, cx)=({}, {}) (H, W)=({}, {})".format(cy, cx, H, W)

            patch_class = list(self._generate_patch_class(img_gray, corner))
            self._draw_patch_class(patch_class, class_idx)

            for patch in patch_class:
                for fern_idx, fern in enumerate(self._ferns):
                    k = fern.calculate(patch)
                    assert 0 <= k < K, "WTF!!!"
                    self._fern_p[fern_idx, class_idx, k] += 1

        Nr = 1

        for fern_idx in iter_timer(range(len(self._ferns)), title="Calculating probs"):
            for cls_idx in range(self._classes_count):
                Nc = np.sum(self._fern_p[fern_idx, cls_idx, :])
                self._fern_p[fern_idx, cls_idx, :] += Nr
                self._fern_p[fern_idx, cls_idx, :] /= (Nc + K * Nr)

                print("P_min={}, P_max={}"
                      .format(np.min(self._fern_p[fern_idx, cls_idx, :]),
                              np.max(self._fern_p[fern_idx, cls_idx, :]))
                      )

        print("Training complete!")

    def match(self, image):
        dims = len(np.shape(image))
        if dims == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        with Timer("track features"):
            corners = self._get_corners(image, self._max_match_corners)

        key_points_trained = []
        key_points_matched = []
        key_points_pairs = []

        most_probable_classes = set()

        for corner in iter_timer(corners, title="Matching corners", print_iterations=False):
            probs = np.zeros((self._classes_count, ))

            patch = self._generate_patch(image, corner)
            for fern_idx, fern in enumerate(self._ferns):
                k = fern.calculate(patch)
                probs += np.log(self._fern_p[fern_idx, :, k])

            most_probable_class = np.argmax(probs)
            most_probable_classes.add(most_probable_class)

            #print("C: {}, p={}".format(corner, np.exp(np.max(probs))))

            key_points_trained.append(self.key_points[most_probable_class])
            key_points_matched.append(corner)
            key_points_pairs.append((self.key_points[most_probable_class], corner))

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

        W = (w+5) * 10
        H = (h+5) * (len(patches) // 10 + 1)

        img = np.zeros((W, H))
        for idx, patch in enumerate(patches):
            x = (idx // 10) * (w + 5)
            y = (idx % 10) * (h + 5)

            img[y:y + h, x: x + w] = patch

        cv2.imwrite("img/train/cls{}.png".format(cls_idx), img)

    def _generate_key_point_pairs(self, n=300):
        pw, ph = self._patch_size

        xs0 = np.random.random_integers(1, pw - 2, n)
        ys0 = np.random.random_integers(1, ph - 2, n)

        xs1 = np.random.random_integers(1, pw - 2, n)
        ys1 = np.random.random_integers(1, ph - 2, n)

        for x0, y0, x1, y1 in zip(xs0, ys0, xs1, ys1):
            yield (y0, x0), (y1, x1)

    def _generate_patch(self, img, center, size=None):
        h, w = np.shape(img)
        h, w = int(h), int(w)

        if size is None:
            ph, pw = self._patch_size
        else:
            ph, pw = size

        assert 0 < pw <= w and 0 < ph <= h

        ph2, pw2 = ph // 2, pw // 2
        y, x = center

        if pw2 <= x and x + pw2 < w and ph2 <= y and y + ph2 <= h:
            # fast way
            return img[int(y) - ph2:int(y) + ph2, int(x) - pw2:int(x) + pw2]




        assert 0 <= y < h and 0 <= x < w, "(y, x)=({}, {}) (h, w)=({}, {})".format(y, x, h, w)

        y, x = int(y) + h, int(x) + w
        x0 = x - pw2
        y0 = y - ph2

        img_extended = cv2.copyMakeBorder(img, h, h, w, w, cv2.BORDER_REFLECT101)

        return img_extended[y0:y0 + ph, x0:x0 + pw]

    def _generate_patch_class(self, img, corner):
        """ generate patch transformations """

        patch = self._generate_patch(img, corner)
        size = np.shape(patch)[:2]

        for _, img in FernDetector._generate_affine_deformations(patch):
            yield img

    @staticmethod
    def _generate_affine_deformations(img, theta_step=5, deformations=20):
        H, W = np.shape(img)[:2]

        center = np.float32(H / 2.0), np.float32(W / 2.0)

        #cv2.line(img, center, (0, 0), 255, 1)

        rotation_matrices = [
            cv2.getRotationMatrix2D((center[1], center[0]), theta, 1.0)
            for theta in range(0, 361)
        ]

        for theta in range(0, 360, theta_step):
            Rt = rotation_matrices[theta]
            N = deformations
            r_phi = np.random.randint(0, 360, N)
            r_lambda1 = np.random.uniform(0.6, 1.5, N)
            r_lambda2 = np.random.uniform(0.6, 1.5, N)
            r_noise_ratio = np.random.uniform(0, 0.1, N)

            for noise_ratio, lambda1, lambda2, phi in zip(r_noise_ratio, r_lambda1, r_lambda2, r_phi):
                Rp  = rotation_matrices[phi]
                Rp1 = rotation_matrices[360 - phi]

                Rl = np.matrix([[lambda1, 0, 0], [0, lambda2, 0]])

                Rz = mult(Rp, mult(Rl, Rp1))

                R = mult(Rt, Rz)
                R_inv = cv2.invertAffineTransform(R)

                warped = cv2.warpAffine(img, R, dsize=(H, W), borderMode=cv2.BORDER_REFLECT101)

                # add gaussian noise
                noise = np.uint8(np.random.normal(0, 25, (W, H)))
                blurred = warped #cv2.GaussianBlur(warped, (7, 7), 2)

                noised = cv2.addWeighted(blurred, 1 - noise_ratio, noise, noise_ratio, 0)

                yield R_inv, noised

    def _get_corners(self, img, max_corners):
        corners = cv2.goodFeaturesToTrack(img, maxCorners=max_corners, qualityLevel=0.01, minDistance=8)

        return ((y, x) for ((x, y), ) in corners)


        img = np.float32(img)
        dst = cv2.cornerHarris(img, 2, 3, 0.04)


        ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
        dst = np.uint8(dst)

        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(img, np.float32(centroids), (5, 5), (-1, -1), criteria)

        return corners[:Z]

        # h, w = np.shape(dst)
        #
        # result = []
        # threshold = dst.max() * 0.01
        # for y in range(h):
        #     for x in range(w):
        #         if dst[y, x] > threshold:
        #             result.append(((x, y), ))
        #
        # return result

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

        for cls_idx in iter_timer(range(self._classes_count), title="Drawing ferns"):
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

                img[y0:y0 + h, x0: x0 + w] = sample # cv2.inpaint(sample, mask, 2, cv2.INPAINT_TELEA)

                # for k in range(K):
                #     p = self._fern_p[fern_idx, k, cls_idx]

            cv2.imwrite("img/learn/cls{}.png".format(cls_idx), img)

    def draw_learned_ferns_2(self, path=""):
        print("Drawing grphics..")
        ferns_count, K, _ = self._fern_p.shape
        x = list(range(K))

        for cls_idx in range(self._classes_count):
            f, axes_arr = plt.subplots(ferns_count,
                                       figsize=(6, 1.5*ferns_count), sharey=True, sharex=True)

            for fern_idx, fern in enumerate(self._ferns):
                axes_arr[fern_idx].plot(x, self._fern_p[fern_idx, :, cls_idx])

            # f.subplots_adjust(hspace=0)

            plt.savefig("{}plot_cls{}.png".format(path, cls_idx), dpi=100)
            plt.close(f)
        print("All graphs were drawn")


if __name__ == "__main__":
    orig = cv2.imread("../samples/sample_ricotta.jpg")
    orig2 = cv2.flip(orig, 1)

    detector = FernDetector(orig)

    detector.draw_learned_ferns()

    kp1, kp2, kp_p = detector.match(orig)

    H, status = cv2.findHomography(kp1, kp2, cv2.RANSAC, 5.0)
    explore_match(orig, orig, kp_p, status=status, H=H)

    if H is not None:
        draw_match_bounds(orig.shape, orig, H)

    # cv2.imshow("orig", cv2.resize(orig, (1024, 768)))

    wait_for_key()
    cv2.destroyAllWindows()


    cam = cv2.VideoCapture("../samples/test_ricotta.avi")
    while True:
        retval, img = cam.read()

        with Timer("matching"):
            kp1, kp2, kp_p = detector.match(img)

        # wait_for_key(13)

        with Timer("homography"):
            H, status = cv2.findHomography(kp1, kp2, cv2.RANSAC, 5.0)

        explore_match(orig, img, kp_pairs=kp_p, status=status, H=H)

        if H is not None:
            draw_match_bounds(img.shape, img, H)
        else:
            print("None :(")

        img = cv2.resize(img, (640, 480))
        cv2.imshow("press any key to continue", img)

        wait_for_key()

        if key_pressed(27):
            break  # esc to quit

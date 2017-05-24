import random
from collections import defaultdict

import cv2
import numpy as np

from asift.common import Timer, iter_timer

from util import wait_for_key, key_pressed, explore_match, COLOR_WHITE
from webcam import draw_match_bounds

Z = 20

class Fern:
    def __init__(self, size, key_point_pairs):
        self._size = size
        self.kp_pairs = key_point_pairs

    def calculate(self, sample):
        """ Calculates feature function on all kp_pairs and generates binary number according to article"""
        result = 0

        for p1, p2 in self.kp_pairs:
            result += int(sample[p1[1]][p1[0]] < sample[p2[1]][p2[0]])

            result <<= 1

        return result

    def draw(self, k, sample):
        levels = []

        for _ in range(len(self.kp_pairs)):
            levels.append(k % 2)
            k >>= 1

        levels.reverse()
        for ((x1, y1), (x2, y2)), level in zip(self.kp_pairs, levels):
            sample[y1, x1] = 255 * level
            sample[y2, x2] = 255 * (1 - level)


class FernDetector:
    def __init__(self, img, patch_size=(16, 16), max_train_corners=20, max_match_corners=200):
        self._patch_size = patch_size
        self._max_train_corners = max_train_corners
        self._max_match_corners = max_match_corners
        self._init_ferns()
        self._train(img)

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

    def _train(self, train_img):
        img_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

        corners = list(self._get_corners(img_gray, self._max_train_corners))
        print(corners)

        img1 = img_gray.copy()
        for corner in corners:
            x, y = corner
            cv2.circle(img1, (x, y), 3, COLOR_WHITE, -1)

        cv2.imshow("corners", img1)
        wait_for_key()

        self._classes_count = len(corners)

        K = 2**(self._S+1)
        self._fern_p = np.zeros((len(self._ferns), self._classes_count, K))
        self.key_points = []

        title = "Training {} classes".format(self._classes_count)
        for class_idx, corner in enumerate(iter_timer(corners, title=title, print_iterations=True)):
            self.key_points.append(corner)

            patch_class = list(self._generate_patch_class(img_gray, corner))
            self._draw_patch_class(patch_class, class_idx)

            for fern_idx, fern in enumerate(self._ferns):
                for patch in patch_class:
                    k = fern.calculate(patch)
                    assert k < K, "WTF!!!"
                    self._fern_p[fern_idx, class_idx, k] += 1

        Nr = 1

        for fern_idx in iter_timer(range(len(self._ferns)), title="Calculating probs"):
            for cls_idx in range(self._classes_count):
                Nc = np.sum(self._fern_p[fern_idx, cls_idx, :])
                self._fern_p[fern_idx, cls_idx, :] += Nr
                self._fern_p[fern_idx, cls_idx, :] /= (Nc + K * Nr)

                # print(max(self._fern_p[fern_idx, cls_idx, :]) - min(self._fern_p[fern_idx, cls_idx, :]))

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

        for corner in iter_timer(corners, title="Matching corners", print_iterations=False):
            probs = np.zeros((self._classes_count, ))

            patch = self._generate_patch(image, corner)
            for fern_idx, fern in enumerate(self._ferns):
                k = fern.calculate(patch)
                probs += np.log(self._fern_p[fern_idx, :, k])

            most_probable_class = np.argmax(probs)

            # print(np.exp(probs))

            key_points_trained.append(self.key_points[most_probable_class])
            key_points_matched.append(corner)
            key_points_pairs.append((self.key_points[most_probable_class], corner))

        return key_points_trained, key_points_matched, key_points_pairs

    def detect(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kp_t, kp_m, kpp = self.match(image)
        H, status = cv2.findHomography(np.array(kp_t), np.array(kp_m), cv2.RANSAC, 5.0)

        h, w = np.shape(image)

        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        if H is not None:
            return np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))

        return []

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
            yield (x0, y0), (pw - x0 - 1, ph - y0 - 1) #(x1, y1)

    def _generate_patch(self, img, center, size=None):
        h, w = np.shape(img)
        h, w = int(h), int(w)

        img_left_mirrored = cv2.flip(img, 1)
        img_top_mirrored = cv2.flip(img, 0)
        img_lt_mirrored = cv2.flip(img_top_mirrored, 1)

        img_extended = np.zeros((h*3, w*3), np.uint8)
        img_extended[0:h,     0:w] = img_lt_mirrored
        img_extended[0:h,     w:w*2] = img_top_mirrored
        img_extended[0:h,     w*2:w*3] = img_lt_mirrored

        img_extended[h:h*2,   0:w] = img_left_mirrored
        img_extended[h:h*2,   w:w*2] = img
        img_extended[h:h*2,   w*2:w*3] = img_left_mirrored

        img_extended[h*2:h*3,  0:w] = img_lt_mirrored
        img_extended[h*2:h*3,  w:w*2] = img_top_mirrored
        img_extended[h*2:h*3,  w*2:w*3] = img_lt_mirrored

        x, y = center
        x, y = int(x) + h, int(y) + w

        if size is None:
            pw, ph = self._patch_size
        else:
            pw, ph = size

        pw2, ph2 = pw // 2, ph // 2

        x0 = x - pw2
        y0 = y - ph2

        return img_extended[y0:y0 + 2 * ph2, x0:x0 + 2 * pw2]

    def _generate_patch_class(self, img, corner):
        """ generate patch transformations """
        w, h = np.shape(img)[:2]
        size = self._patch_size[1]*4, self._patch_size[0]*4
        patch = self._generate_patch(img, corner, size)

        cx, cy = size
        cx, cy = cx // 2, cy // 2

        center = np.float32(cx), np.float32(cy)

        rotation_matrices = [
            cv2.getRotationMatrix2D(center, theta, 1.0)
            for theta in range(0, 361)
        ]

        pw, ph = self._patch_size

        for theta in range(0, 360, 360):
            Rt = rotation_matrices[theta]
            N = 20
            r_phi = [0] #np.random.randint(0, 360, N)
            r_lambda1 = [1.0] #np.random.uniform(0.9, 1.1, N)
            r_lambda2 = [1.0] #np.random.uniform(0.9, 1.1, N)

            for lambda1, lambda2, phi in zip(r_lambda1, r_lambda2, r_phi):
                Rp  = rotation_matrices[phi]
                Rp1 = rotation_matrices[360 - phi]

                Rl = np.matrix([[lambda1, 0, 0], [0, lambda2, 0]])

                warped = cv2.warpAffine(patch, Rp1, dsize=size)
                warped = cv2.warpAffine(warped, Rl, dsize=size)
                warped = cv2.warpAffine(warped, Rp, dsize=size)
                warped = cv2.warpAffine(warped, Rt, dsize=size)

                # add gaussian noise
                #noise = np.uint8(np.random.normal(0, 25, (size[1], size[0])))
                blurred = warped #cv2.GaussianBlur(warped, (7, 7), 2)

                noise_ratio = 0

                noised = blurred # cv2.addWeighted(blurred, 1 - noise_ratio, noise, noise_ratio, 0)

                x0 = int(cx - pw / 2)
                y0 = int(cy - ph / 2)

                yield noised[y0:y0 + ph, x0:x0 + pw]

        # params = [(1.0, 0.0)]
        # t = 1.0
        # for phi in np.arange(-90, 90, 12.0 / t):
        #     params.append((t, phi))
        #
        # for (t, phi) in params:
        #     patch1, mask, Ai = affine_skew(t, phi, patch)
        #     yield patch1

    def _get_corners(self, img, max_corners):
        corners = cv2.goodFeaturesToTrack(img, maxCorners=max_corners, qualityLevel=0.01, minDistance=8)

        return (corner for (corner, ) in corners)


        img = np.float32(img)
        dst = cv2.cornerHarris(img, 2, 3, 0.04)


        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
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
            for (x1, y1), (x2, y2) in fern.kp_pairs:
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


if __name__ == "__main__":
    orig = cv2.imread("../samples/sample_ricotta.jpg")
    orig2 = cv2.flip(orig, 1)

    detector = FernDetector(orig)

    detector.draw_learned_ferns()

    kp1, kp2, kp_p = detector.match(orig)

    img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    H, status = cv2.findHomography(np.array(kp1), np.array(kp2), cv2.RANSAC, 5.0)
    explore_match(img, img, kp_p, status=status, H=H)

    if H is not None:
        draw_match_bounds(orig.shape, orig, H)

    # cv2.imshow("orig", cv2.resize(orig, (1024, 768)))

    wait_for_key()
    cv2.destroyAllWindows()

    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    cam = cv2.VideoCapture("../samples/test_ricotta.avi")
    while True:
        retval, img = cam.read()

        with Timer("matching"):
            kp1, kp2, kp_p = detector.match(img)

        # wait_for_key(13)

        with Timer("homography"):
            H, status = cv2.findHomography(np.array(kp1), np.array(kp2), cv2.RANSAC, 5.0)

        explore_match(orig, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), kp_p, status, H)

        if H is not None:
            draw_match_bounds(img.shape, img, H)
        else:
            print("None :(")

        img = cv2.resize(img, (640, 480))
        cv2.imshow("press any key to continue", img)

        wait_for_key()

        if key_pressed(27):
            break  # esc to quit

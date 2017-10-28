import cv2
import numpy as np
import operator
from util import iter_timer, flip_points, mult, Timer, wait_for_key


def get_corners(img, max_corners):
    corners = cv2.goodFeaturesToTrack(img, maxCorners=max_corners, qualityLevel=0.01, minDistance=8)
    return ((y, x) for ((x, y),) in corners)


def get_stable_corners(train_img, max_corners=100):
    H, W = np.shape(train_img)[:2]

    CORNER_CNT = 500

    corners = list(get_corners(train_img, CORNER_CNT))

    cv2.imshow("original", train_img)

    with Timer("Corner generation"):
        for R_inv, img in generate_deformations(train_img, theta_step=36, deformations=3):
            new_corners = np.array(list(get_corners(img, CORNER_CNT)), dtype=np.float32)

            # y,x --> x,y
            new_corners = flip_points(new_corners)

            t = [[1]] * len(new_corners)

            new_corners = np.transpose(np.hstack((new_corners, t)))

            corners_inv = np.transpose(np.dot(R_inv, new_corners)) #cv2.transform(new_corners, R_inv)
            # corners_inv = flip_points(corners_inv)

            corners.extend(corners_inv)

    corners = sorted(corners, key=lambda p: p[0])

    collectors = []

    def find_best_collector(point):
        threshold = 2 * 2
        x, y = point

        l = len(collectors)

        for idx in reversed(range(l)):
            cx, cy, _ = collectors[idx]

            xdist = abs(cx - x)
            dist2 = xdist ** 2 + (cy - y) ** 2
            if dist2 <= threshold:
                return idx

            # collectors are sorted by x
            # if xdist > threshold then for all remaining collectors dist2 > threshold
            if xdist > threshold:
                break

        return None

    skip_count = 0

    for cx, cy in iter_timer(corners, "Stable corner detection", print_iterations=False):
        if not (0 <= cy < H and 0 <= cx < W):
            skip_count += 1
            continue

        best_collector = find_best_collector((cx, cy))
        if best_collector is None:
            collectors.append((cx, cy, 1))
        else:
            x, y, cnt = collectors[best_collector]
            collectors[best_collector] = ((x * cnt + cx) / (cnt + 1), (y * cnt + cy) / (cnt + 1), cnt + 1)

    collectors = sorted(collectors, key=lambda c: -c[2])

    print("Skipped {}".format(skip_count))

    for x, y, _ in collectors[:max_corners]:
        yield int(y), int(x)


def generate_deformations(img, theta_step=1, deformations=30):
    H, W = np.shape(img)[:2]

    center = np.float32(H / 2.0), np.float32(W / 2.0)

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
            Rp = rotation_matrices[phi]
            Rp1 = rotation_matrices[360 - phi]

            Rl = np.matrix([[lambda1, 0, 0], [0, lambda2, 0]])

            Rz = mult(Rp, mult(Rl, Rp1))

            R = mult(Rt, Rz)
            R_inv = cv2.invertAffineTransform(R)

            warped = cv2.warpAffine(img, R, dsize=(H, W), borderMode=cv2.BORDER_REFLECT101)

            # add gaussian noise
            noise = np.uint8(np.random.normal(0, 25, (W, H)))
            blurred = warped  # cv2.GaussianBlur(warped, (7, 7), 25)

            noised = cv2.addWeighted(blurred, 1 - noise_ratio, noise, noise_ratio, 0)

            yield R_inv, noised
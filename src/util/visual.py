import cv2
import numpy as np
from util import iter_timer, flip_points, mult


def get_corners(img, max_corners):
    corners = cv2.goodFeaturesToTrack(img, maxCorners=max_corners, qualityLevel=0.01, minDistance=8)
    return ((y, x) for ((x, y),) in corners)


def get_stable_corners(train_img, max_corners=100):
    H, W = np.shape(train_img)[:2]

    def find_collector(collectors, point, radius=3):
        radius2 = radius ** 2
        point = np.array(point)

        best_collector = None
        for collector in collectors:
            dist = collector.dist2(point)
            if dist <= radius2:
                best_collector = collector
                radius2 = dist

        return best_collector

    corners = list(get_corners(train_img, 500))
    for R_inv, img in iter_timer(generate_deformations(train_img, theta_step=36, deformations=3), "Corner generation"):
        new_corners = np.array([list(get_corners(img, 500))])

        (corners_inv,) = cv2.transform(new_corners, R_inv)
        corners_inv = np.array(flip_points(corners_inv), dtype=int)

        corners.extend(corners_inv)

    collectors = {}
    for cx, cy in iter_timer(corners, "Stable corner detection"):
        corner = int(cx), int(cy)
        if not (0 <= cy < H and 0 <= cx < W):
            continue

        collectors[corner] = collectors.get(corner, 0) + 1

    collectors = sorted(collectors.items(), key=lambda c: -c[1])

    for (x, y), _ in collectors[:max_corners]:
        yield y, x


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
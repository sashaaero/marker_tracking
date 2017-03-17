import cv2
import numpy as np

from asift.asift import affine_skew, get_camera_params


class FernDetector:
    def __init__(self, img, path_size=(32, 32)):
        self._train(img)
        self._path_size = path_size

    def _train(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners = cv2.goodFeaturesToTrack(img_gray, maxCorners=500, qualityLevel=0.01, minDistance=16)

        h, w = np.shape(img_gray)

        pw, ph = self._path_size
        pw2 = pw // 2 + 1
        ph2 = ph // 2 + 1

        params = get_camera_params()
        for ((x, y), ) in corners:
            # ensure path is inside image
            cx = min(max(x, pw2), w - 1 - pw2)
            cy = min(max(y, ph2), h - 1 - ph2)

            # top left corner
            x0, y0 = cx - pw2, cy - ph2

            patch = img_gray[y0:y0 + ph, x0:x0 + pw]








class FernMatcher:
    pass


if __name__ == "__main__":
    orig = cv2.imread("../sample.jpg")
    FernDetector(orig)

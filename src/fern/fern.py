import cv2
import numpy as np

from asift.asift import affine_skew


class FernDetector:
    def __init__(self, img):
        self._train(img)

    def _train(self, img):
        pass


class FernMatcher:
    pass


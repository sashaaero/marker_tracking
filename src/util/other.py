import numpy as np
import cv2


def mult(m1, m2):
    """Multiplies two affine transforamtion matrces"""
    m1_temp = np.vstack((m1, [0, 0, 1]))
    m2_temp = np.vstack((m2, [0, 0, 1]))
    result = m1_temp * m2_temp

    return result[:2, :]


def flip_points(a):
    a = np.array(a)
    return np.flip(a, 1)


def transform32(points, H, add=(0, 0)):
    """Transform list of points [(x, y), ...]
        
        :return H * points.T() + add
    """
    points = np.float32(points)
    return np.int32(cv2.perspectiveTransform(points.reshape(1, -1, 2), H).reshape(-1, 2) + add)

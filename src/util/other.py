import numpy as np


def mult(m1, m2):
    """Multiplies two affine transoframtion matrces"""
    m1_temp = np.vstack((m1, [0, 0, 1]))
    m2_temp = np.vstack((m2, [0, 0, 1]))
    result = m1_temp * m2_temp

    return result[:2, :]


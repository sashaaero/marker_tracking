import numpy as np
import cv2
import os


def get_ground_truth(dataset):
    """ Parses input file and returns list of pattern corners clockwise"""
    result = []

    with open("samples/{}/groundtruth_rect.txt".format(dataset), 'r') as f:
        for line in f.readlines():
            x, y, w, h = line.split()
            x, y, w, h = int(x), int(y), int(w), int(h)
            result.append([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    return result


def get_sample(dataset):
    (x, y), _, (x1, y1), _ = get_ground_truth(dataset)[0]
    sample = cv2.imread("samples/{}/img/0001.jpg".format(dataset))

    return sample[y:y1, x:x1]


def get_images(dataset):
    for file in os.listdir("samples/{}/img".format(dataset)):
        yield cv2.imread("samples/{}/img/{}".format(dataset, file))

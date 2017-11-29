from typing import TextIO

import cv2

import fern
import util
import numpy as np


def calc_metric(orig, points):
    """
    See Planar Object Tracking in the Wild A Benchmark, p. 6
    :param orig:
    :param points:
    :return:
    """
    result = 0
    for (xo, yo), (xp, yp) in zip(orig, points):
        result += (xo - xp) ** 2 + (yo - yp) ** 2

    return np.sqrt(result) / 2


def benchmark_dataset(detector,
                      video,
                      frame_flags: TextIO,
                      gt_homography: TextIO,
                      gt_points: TextIO):
    # drop the first line of data
    next(frame_flags), next(gt_homography), next(gt_points)

    result = []
    for idx, (frame, flag, Hline, Pline) in \
            enumerate(zip(util.get_frames(video), frame_flags, gt_homography, gt_points)):

        x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, Pline.strip().split()))
        flag = int(flag.strip())
        print("{}, ".format(idx), end="")

        if idx % 2 == 1 or flag > 0:
            print("dropped")
            continue

        points, H = detector.detect(frame)
        metric = calc_metric([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], points)

        print("m = {}".format(metric))

        result.append(metric)

    return result


def train_detector(video, gt_points: TextIO):
    frame = next(util.get_frames(video))
    gt_points = np.array(list(util.grouper(map(float, next(gt_points).strip().split()), 2)))

    sample_corners = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], dtype=np.float32)

    H, _ = cv2.findHomography(gt_points, sample_corners, cv2.RANSAC, 5.0)
    sample = cv2.warpPerspective(frame, H, (640, 480))

    return fern.FernDetector.train(sample, max_train_corners=20, max_match_corners=500)


if __name__ == "__main__":
    with open("../../datasets/annotation/V01_1_flag.txt", 'r') as flag, \
         open("../../datasets/annotation/V01_1_gt_homography.txt", 'r') as homography, \
         open("../../datasets/annotation/V01_1_gt_points.txt", 'r') as points:

        video = cv2.VideoCapture("../../datasets/V01/V01_1.avi")
        detector = train_detector(video, points)

        # reset video an points file positions
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        points.seek(0)

        result = benchmark_dataset(
            detector=detector,
            video=video,
            frame_flags=flag,
            gt_homography=homography,
            gt_points=points)

        print(result)
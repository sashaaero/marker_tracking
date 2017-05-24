import util
import cv2
import fern
import numpy as np


def benchmark_dataset(dataset):
    ground_truth = util.get_ground_truth(dataset)
    sample = util.get_sample(dataset)

    detector = fern.FernDetector(sample, max_train_corners=50, max_match_corners=200)
    detector.draw_learned_ferns()

    img = sample.copy()
    detection_box = detector.detect(img)
    examine_detection(detector, sample, img, ground_truth[0], detection_box)

    for truth_box, img in zip(ground_truth, util.get_images(dataset)):
        detection_box = detector.detect(img)
        if len(detection_box) == 0:
            print("Homography not found")

        examine_detection(detector, sample, img, truth_box, detection_box)


def examine_detection(detector, sample, img, truth_box, detection_box):
    util.draw_poly(img, truth_box, color=util.COLOR_WHITE)
    util.draw_poly(img, detection_box, color=util.COLOR_RED)

    cv2.imshow("step", img)

    kp_t, kp_m, kp_p = detector.match(img)
    H, status = cv2.findHomography(np.array(kp_t), np.array(kp_m), cv2.RANSAC, 5.0)
    util.explore_match(sample, img, kp_pairs=kp_p, H=H, status=status)

    util.wait_for_key()

if __name__ == "__main__":
    # benchmark_dataset("ClifBar")
    benchmark_dataset("Box")
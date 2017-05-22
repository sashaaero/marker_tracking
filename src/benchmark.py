import util
import cv2
import fern


def benchmark_clifbar():
    dataset = "ClifBar"

    ground_truth = util.get_ground_truth(dataset)
    sample = util.get_sample(dataset)

    detector = fern.FernDetector(sample)
    detector.draw_learned_ferns()

    for truth_box, img in zip(ground_truth, util.get_images(dataset)):
        detection_box = detector.detect(img)
        if not detection_box:
            print("Homography not found")

        util.draw_poly(img, truth_box, color=util.COLOR_WHITE)
        util.draw_poly(img, detection_box, color=util.COLOR_RED)

        cv2.imshow("step", img)
        util.wait_for_key()


if __name__ == "__main__":
    benchmark_clifbar()
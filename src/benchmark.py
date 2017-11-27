import cv2
import fern
import random
import util


def benchmark_dataset(dataset, explore=True):
    ground_truth = util.get_ground_truth(dataset)
    sample = util.get_sample(dataset)

    detector = fern.FernDetector(sample, max_train_corners=20, max_match_corners=500)

    img = sample.copy()
    detection_box, _ = detector.detect(img)
    util.examine_detection(detector, sample, img, ground_truth[0], detection_box, explore=explore)

    for truth_box, img in zip(ground_truth, util.get_images(dataset)):
        detection_box, _ = detector.detect(img)
        if len(detection_box) == 0:
            print("Homography not found")

        util.examine_detection(detector, sample, img, truth_box, detection_box, explore=explore)


def benchmark_sample(deserialize=False):
    cam = cv2.VideoCapture("samples/test_ricotta.avi")
    sample = cv2.imread("samples/sample_ricotta.jpg")

    serialization_path = "samples/ricotta_detector.dat"

    if not deserialize:
        detector = fern.FernDetector.train(sample, max_train_corners=50, max_match_corners=500)

        with open(serialization_path, 'w') as f:
            detector.serialize(f)

    else:
        with open(serialization_path, 'r') as f:
            detector = fern.FernDetector.deserialize(f)

    detection_box, _ = detector.detect(sample)
    if len(detection_box) == 0:
        print("Homography not found")

    # examine_detection(detector, sample, sample, [], detection_box, explore=True)

    while True:
        ret, img = cam.read()
        if not ret:
            break

        detection_box, _ = detector.detect(img)
        if len(detection_box) == 0:
            print("Homography not found")

        util.examine_detection(detector, sample, img, [], detection_box, explore=True)

    cam.release()


if __name__ == "__main__":
    random.seed(1234)
    # benchmark_sample()
    benchmark_sample(deserialize=True)
    # benchmark_dataset("ClifBar")
    # benchmark_dataset("Box")

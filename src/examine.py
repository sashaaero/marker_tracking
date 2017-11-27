import cv2
import fern
import util


def examine(cam, detector, sample):
    while True:
        ret, img = cam.read()
        if not ret:
            break

        detection_box, _ = detector.detect(img)
        if len(detection_box) == 0:
            print("Homography not found")

        util.examine_detection(detector, sample, img, [], detection_box, explore=True)


if __name__ == "__main__":
    cam = cv2.VideoCapture("samples/test_ricotta.avi")
    sample = cv2.imread("samples/sample_ricotta.jpg")
    serialization_path = "samples/ricotta_detector.dat"
    with open(serialization_path, 'r') as f:
        detector = fern.FernDetector.deserialize(f)

    examine(cam, detector, sample)

    cam.release()

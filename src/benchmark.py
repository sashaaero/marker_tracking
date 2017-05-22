import util
import cv2


def benchmark_clifbar():
    dataset = "ClifBar"

    ground_truth = util.get_ground_truth(dataset)

    for box, img in zip(ground_truth, util.get_images(dataset)):
        util.draw_poly(img, box)

        cv2.imshow("step", img)

        util.wait_for_key()




if __name__ == "__main__":
    benchmark_clifbar()
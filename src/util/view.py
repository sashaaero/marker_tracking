import cv2
import numpy as np


COLOR_WHITE = (255, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)


def capture_img(cam, size=(640, 480)):
    _, img = cam.read()
    return cv2.resize(img, size)


def capture_orig(cam):
    while True:
        img = capture_img(cam)

        cv2.imshow("Press enter to capture", cv2.flip(img, 1))
        if cv2.waitKey(1) == 13:
            return img


def key_pressed(key=None, wait=1):
    pressed = cv2.waitKey(wait)
    if key is None:
        return 0 <= pressed < 255
    else:
        return pressed == key


def wait_for_key(key=None):
    while not key_pressed(key):
        pass


def draw_poly(img, corners, color=COLOR_WHITE):
    corners = [np.int32(corners)]
    cv2.polylines(img, corners, True, color)


def explore_match(sample, match, kp_pairs, window_name="Match exploration", status=None, H=None):
    h1, w1 = sample.shape[:2]
    h2, w2 = match.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2, 3), np.uint8)
    vis[:h1, :w1, :] = sample
    vis[:h2, w1:w1+w2, :] = match
    # vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0]))
        p2.append(np.int32(np.array(kpp[1]) + [w1, 0]))

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = COLOR_GREEN
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = COLOR_RED
            r = 2
            thickness = 5
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if not inlier:
            cv2.line(vis, (x1, y1), (x2, y2), (128, 128, 255))

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), COLOR_GREEN)

    cv2.imshow(window_name, vis)

    return vis


import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    frame_captured, frame = cap.read()
    if frame_captured:
        frame = cv2.flip(frame, 1)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.imwrite("sample_emc.jpg", frame)
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()


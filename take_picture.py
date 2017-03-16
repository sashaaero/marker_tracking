import cv2

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame, 1)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("sample_gaga.jpg", frame)
            break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()


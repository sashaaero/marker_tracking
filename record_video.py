import cv2

cap = cv2.VideoCapture(1)

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('test.avi', -1, 20.0, (640,480))

recording = False
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame, 1)

        if recording:
            # write the flipped frame
            out.write(frame)

        cv2.imshow('frame',frame)

        key = cv2.waitKey(1)
        if key == 13:
            recording = True
        if key & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()


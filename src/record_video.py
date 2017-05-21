import cv2

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('test_ricotta.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (640,480))

recording = False
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        if recording:
            # write the flipped frame
            print("writing")
            out.write(frame)

        cv2.imshow('press ENTER to start recording, q to exit',frame)

        key = cv2.waitKey(1)
        if key == 13:
            recording = True
            print("Start recording..")
        if key & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()


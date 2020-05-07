import cv2
import pyautogui

capture = cv2.VideoCapture(0)
cascade_path = "./haarcascades/haarcascade_frontalface_alt_tree.xml"

w = int(pyautogui.size().width/2)
h = int(pyautogui.size().height/2)
min_detect_size = 50
ret, frame = capture.read()
cv2.imshow('title',frame)
cv2.moveWindow('title', 0, 0)

while(True):
    ret, frame = capture.read()
    # resize the window
    windowsize = (w, h)
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, windowsize)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=2, minSize=(min_detect_size, min_detect_size))
    color = (255, 255, 255)

    if len(facerect) > 0:
        for rect in facerect:
            cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color, thickness=2)
    cv2.imshow('title',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
import cv2

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if ret :
        cv2.imshow("My first cam", frame)
        cv2.imshow("Second", frame)
    key = cv2.waitKey(1)

    if ord("q") == key & 0xff:
        break

    if ord("c") == key & 0xff:
        cv2.imwrite("classroom.png", frame)



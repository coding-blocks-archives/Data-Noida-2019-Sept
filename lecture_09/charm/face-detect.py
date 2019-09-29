import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_flat = None

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_list = []

while True:

    ret, frame = cap.read()
    if ret :
        cv2.imshow("My first cam", frame)

        faces = classifier.detectMultiScale(frame)

        if len(faces) > 0:

            areas = np.product(np.array(faces)[:, 2:], axis=1)
            index = areas.argmax()
            face = faces[index]
            x, y, w, h = face

            cut = frame[y:y+h, x:x+w]
            cut = cv2.resize(cut, (100, 100))

            gray = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
            face_flat = gray.flatten()

            cv2.imshow("jo bhi naam", cut)

    key = cv2.waitKey(1)

    if ord("q") == key & 0xff:
        break

    if ord("c") == key & 0xff:
        if len(faces) > 0:
            face_list.append(face_flat)



X = np.array(face_list)

print(X.shape)






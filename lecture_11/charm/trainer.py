import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

person_name = input("Enter your name")

face_list = []
face_count = 10

while(True):

    ret, frame = cap.read()
    faces = None

    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(gray)
        if len(faces) > 0:
            areas = np.product(np.array(faces)[:, 2:], axis=1)
            best_index = areas.argmax()
            chosen_face = faces[best_index]
            x, y, w, h = chosen_face
            img_cut = gray[y:y+h, x:x+w]
            face = cv2.resize(img_cut, (100, 100))
            cv2.imshow("blah", face)


    key = cv2.waitKey(1)

    if key & 0xff == ord('q'):
        break

    if key & 0xff == ord('c'):
        if len(faces) > 0:
            face_list.append(face.flatten())
            face_count -= 1
            print("remaining", face_count)
            if face_count == 0:
                break

X = np.array(face_list)
y = np.full((len(X), 1), person_name)

data = np.hstack([y, X])

if os.path.exists("face_data.npy"):
    old = np.load("face_data.npy")
    data = np.vstack([old, data])

data.dump("face_data.npy")

print(data.shape)



cap.release()
cv2.destroyAllWindows()

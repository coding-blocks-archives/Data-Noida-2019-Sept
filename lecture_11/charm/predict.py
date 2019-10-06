import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data = np.load("face_data.npy", allow_pickle=True)

X = data[:, 1:].astype(int)
y = data[:, 0]

model = KNeighborsClassifier(3)
model.fit(X, y)

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):

    ret, frame = cap.read()
    faces = None

    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(gray)
        for chosen_face in faces:
            x, y, w, h = chosen_face
            img_cut = gray[y:y+h, x:x+w]
            face = cv2.resize(img_cut, (100, 100))
            flat = face.flatten()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0))
            name = model.predict([flat])
            frame = cv2.putText(frame, str(name[0]), (x+30, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    cv2.imshow("window", frame)
    key = cv2.waitKey(1)

    if key & 0xff == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()



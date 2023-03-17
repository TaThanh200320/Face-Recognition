import cv2 as cv
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import pickle as pkl
from keras_facenet import FaceNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
facenet = FaceNet()
faces_embeddings = np.load("Faces_embedded_2class.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pkl.load(open("svm_model.pkl", 'rb'))

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 3)
    face_imgs = []
    for x, y, w, h in faces:
        face_img = frame[y:y+h, x:x+w]
        face_imgs.append(cv.resize(face_img, (160, 160)))
    if len(face_imgs) > 0:
        face_imgs = np.stack(face_imgs)
        face_embeddings = facenet.embeddings(face_imgs)
        face_names = encoder.inverse_transform(model.predict(face_embeddings))
    for i, (x, y, w, h) in enumerate(faces):
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 5)
        cv.putText(frame, str(face_names[i]), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)
    cv.imshow("Face Recognition", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

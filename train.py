import os
import cv2
import joblib
import numpy as np
from sklearn.svm import SVC
from utils import detect_face, preprocess, extract_hog

X = []
y = []
labels = {}

dataset_path = "dataset"
label_id = 0

for label_id, person_name in enumerate(os.listdir(dataset_path)):
    person_path = os.path.join(dataset_path, person_name)
    labels[label_id] = person_name

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)

        face, _ = detect_face(image)
        if face is None:
            continue

        face = preprocess(face)
        features = extract_hog(face)

        X.append(features)
        y.append(label_id)

X = np.array(X)
y = np.array(y)

model = SVC(kernel="linear", probability=True)
model.fit(X, y)

joblib.dump(model, "face_model.pkl")
joblib.dump(labels, "labels.pkl")

print("Модель обучена и сохранена в face_model.pkl")

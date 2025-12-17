import cv2
import joblib
import numpy as np
from utils import detect_face, preprocess, extract_hog

model = joblib.load("face_model.pkl")
labels = joblib.load("labels.pkl")

image = cv2.imread("test.jpg")
if image is None:
    raise FileNotFoundError("Файл test.jpg не найден!")

face, box = detect_face(image)
if face is None or box is None:
    print("Лицо не найдено")
    exit()

face = preprocess(face)
features = extract_hog(face)

probs = model.predict_proba([features])[0]
best_id = int(np.argmax(probs))
confidence = float(probs[best_id] * 100)
name = labels[best_id]

x, y, w, h = map(int, box)

if confidence >= 70:
    text = f"Добро пожаловать, {name}! ({confidence:.1f}%)"
    color = (0, 255, 0)
else:
    text = "Доступ запрещён"
    color = (0, 0, 255)

cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
cv2.putText(
    image,
    text,
    (x, y - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    color,
    2
)

cv2.imshow("Face Access System", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

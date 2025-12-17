import os
import cv2
import joblib
import numpy as np
from utils import detect_face, preprocess, extract_hog

model = joblib.load("face_model.pkl")
labels = joblib.load("labels.pkl")

test_folder = "test_dataset"
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Не удалось открыть файл {img_name}")
        continue

    face, box = detect_face(image)
    if face is None or box is None:
        print(f"Лицо не найдено на {img_name}")
        continue

    face = preprocess(face)
    features = extract_hog(face)

    probs = model.predict_proba([features])[0]
    best_id = int(np.argmax(probs))
    confidence = float(probs[best_id] * 100)
    name = labels[best_id]

    x, y, w, h = map(int, box)
    if confidence >= 30:
        text = f"Hi, {name}! ({confidence:.1f}%)"
        color = (0, 255, 0)
    else:
        text = "Error"
        color = (0, 0, 255)

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_path = os.path.join(output_folder, img_name)
    cv2.imwrite(output_path, image)
    print(f"Результат сохранён: {output_path}")

print("Обработка завершена!")

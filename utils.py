import cv2
from skimage.feature import hog

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    return face, (x, y, w, h)

def preprocess(face):
    face = cv2.resize(face, (128, 128))
    face = cv2.equalizeHist(face)
    return face

def extract_hog(face):
    return hog(
        face,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

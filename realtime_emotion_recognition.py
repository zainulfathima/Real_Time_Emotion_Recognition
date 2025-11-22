import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "emotion_model.h5"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
IMG_SIZE = 48
EMOTION_LABELS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

def preprocess(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r = cv2.resize(g,(IMG_SIZE,IMG_SIZE))
    n = r.astype('float32')/255.0
    return np.expand_dims(n,axis=(0,-1))

model = load_model(MODEL_PATH)
cascade = cv2.CascadeClassifier(CASCADE_PATH)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        f = frame[y:y+h, x:x+w]
        p = preprocess(f)
        pred = model.predict(p)
        idx = np.argmax(pred)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame,EMOTION_LABELS[idx],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
    cv2.imshow("Emotion",frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

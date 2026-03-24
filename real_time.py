# realtime.py
# Opens your webcam, detects faces in every frame,
# runs the trained CNN on each face, and overlays the emotion label.

import cv2
import numpy as np
import tensorflow as tf

# ── 1. SETUP ──────────────────────────────────────────────
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy',
            'Neutral', 'Sad', 'Surprise']

# Color per emotion (BGR format for OpenCV)
EMOTION_COLORS = {
    'Angry':    (0, 0, 220),
    'Disgust':  (0, 140, 0),
    'Fear':     (130, 0, 200),
    'Happy':    (0, 210, 255),
    'Neutral':  (180, 180, 180),
    'Sad':      (200, 100, 0),
    'Surprise': (0, 180, 255)
}

# Load trained model and face detector
print("Loading model...")
model = tf.keras.models.load_model('emotion_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ── 2. OPEN WEBCAM ────────────────────────────────────────
cap = cv2.VideoCapture(0)   # 0 = default webcam
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Webcam running. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── 3. FACE DETECTION ─────────────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # haar works on grayscale

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,    # how much image is reduced at each scale
        minNeighbors=5,     # how many neighbors each candidate must have
        minSize=(30, 30)    # ignore very tiny detections
    )

    # ── 4. PREDICT EMOTION FOR EACH FACE ──────────────────
    for (x, y, w, h) in faces:
        # Crop the face region
        roi_gray = gray[y:y+h, x:x+w]

        # Resize to 48x48 (what the model expects)
        roi_resized = cv2.resize(roi_gray, (48, 48))

        # Normalize and reshape: (48,48) → (1, 48, 48, 1)
        roi_input = roi_resized.astype('float32') / 255.0
        roi_input = np.expand_dims(roi_input, axis=0)    # add batch dim
        roi_input = np.expand_dims(roi_input, axis=-1)   # add channel dim

        # Model prediction → array of 7 probabilities
        predictions = model.predict(roi_input, verbose=0)[0]
        emotion_idx  = np.argmax(predictions)
        emotion_label = EMOTIONS[emotion_idx]
        confidence    = predictions[emotion_idx] * 100

        # ── 5. DRAW RESULTS ON FRAME ──────────────────────
        color = EMOTION_COLORS[emotion_label]

        # Bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Label background box (so text is readable)
        label_text = f"{emotion_label} ({confidence:.1f}%)"
        (tw, th), _ = cv2.getTextSize(label_text,
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y-th-10), (x+tw+6, y), color, -1)

        # Emotion label text
        cv2.putText(frame, label_text,
                    (x+3, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        # ── 6. MINI PROBABILITY BAR CHART ─────────────────
        # Shows all 7 emotion scores as horizontal bars
        bar_x, bar_y = 10, 10
        for i, (emo, prob) in enumerate(zip(EMOTIONS, predictions)):
            bar_w = int(prob * 150)
            emo_color = EMOTION_COLORS[emo]
            cv2.rectangle(frame,
                          (bar_x, bar_y + i*22),
                          (bar_x + bar_w, bar_y + i*22 + 16),
                          emo_color, -1)
            cv2.putText(frame, f"{emo[:3]} {prob*100:.0f}%",
                        (bar_x + bar_w + 4, bar_y + i*22 + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (220, 220, 220), 1)

    # ── 7. DISPLAY FRAME ──────────────────────────────────
    cv2.imshow('Emotion Detector — Press Q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
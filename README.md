# Emotion-Detector
**A deep learning project that detects 7 human emotions in real-time from your webcam** using a Convolutional Neural Network (CNN) trained from scratch on the FER2013 dataset.

**Emotions detected:**  
**Angry • Disgust • Fear • Happy • Neutral • Sad • Surprise**

---

## ✨ Features

- Custom CNN built from scratch (no transfer learning)
- Real-time webcam emotion detection with confidence scores
- Live probability bar chart for all 7 emotions
- Data augmentation for better generalization
- Automatic model saving with checkpoints
- Training visualization (accuracy & loss curves)

---

## 🛠️ Tech Stack

- **TensorFlow / Keras** – Deep learning framework
- **OpenCV** – Webcam & face detection
- **NumPy + Matplotlib** – Data handling & visualization
- **Scikit-learn** – Evaluation (optional)
- **FER2013 Dataset** – 35,000+ labeled face images

---

## 📁 Project Structure
```
emotion_detector/
├── dataset/                          # FER2013 dataset
│   ├── train/
│   └── test/
├── model.py                          # CNN architecture
├── train_model.py                    # Training script
├── realtime.py                       # Live webcam detector
├── emotion_model.h5                  # Trained model (after training)
├── haarcascade_frontalface_default.xml
├── training_plot.png                 # Training graphs
└── README.md
```


---

## 🚀 Step-by-Step Setup

### Step 1 — Create Project & Install Dependencies

```bash
mkdir emotion_detector
cd emotion_detector

pip install tensorflow numpy opencv-python matplotlib scikit-learn kaggle
```

### Step 2 — Download FER2013 Dataset

Go to: https://www.kaggle.com/datasets/msambare/fer2013
Sign in → Click Download
Extract the zip and place the train/ and test/ folders inside a dataset/ folder in your project root.

Your folder should look like this:
```
dataset/
├── train/
│   ├── angry/   ├── disgust/   ├── fear/
│   ├── happy/   ├── neutral/   ├── sad/   └── surprise/
└── test/  (same subfolders)
```

### Step 3 — Train the Model
Run: python train_model.py

### Step 4 — Get Haar Cascade for Face Detection
Run: 
```
python -c "import cv2, shutil; src = cv2.__file__.replace('__init__.py','data/haarcascade_frontalface_default.xml'); shutil.copy(src, '.')"
```

### Step 5 — Run Real-Time Emotion Detector
Run: python real_time.py

Press Q to quit

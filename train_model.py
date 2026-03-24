# train_model.py
# This script loads the dataset, preprocesses it,
# trains the CNN, and saves the trained weights.

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from model import build_emotion_model

# ── 1. PATHS ──────────────────────────────────────────────
TRAIN_DIR = 'data/train'
TEST_DIR  = 'data/test'
IMG_SIZE  = 48        # FER2013 images are 48x48
BATCH     = 64
EPOCHS    = 50        # model will stop early if it stops improving

# ── 2. DATA GENERATORS (with augmentation) ────────────────
# Augmentation artificially multiplies training data
# by randomly flipping and shifting images
train_datagen = ImageDataGenerator(
    rescale=1./255,           # normalize pixel values 0-255 → 0-1
    horizontal_flip=True,     # randomly mirror faces
    width_shift_range=0.1,    # randomly shift left/right by 10%
    height_shift_range=0.1,   # randomly shift up/down by 10%
    zoom_range=0.1,           # random small zoom
    rotation_range=10         # slight rotations
)

test_datagen = ImageDataGenerator(rescale=1./255)  # no augmentation on test

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',    # convert to grayscale (1 channel)
    batch_size=BATCH,
    class_mode='categorical'   # one-hot labels
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH,
    class_mode='categorical'
)

# ── 3. BUILD MODEL ────────────────────────────────────────
model = build_emotion_model(num_classes=7)
model.summary()   # prints layer shapes — helpful for debugging

# ── 4. COMPILE ────────────────────────────────────────────
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',   # standard for multi-class
    metrics=['accuracy']
)

# ── 5. CALLBACKS ──────────────────────────────────────────
callbacks = [
    # Save only the best model weights (by val_accuracy)
    ModelCheckpoint('emotion_model.h5', monitor='val_accuracy',
                    save_best_only=True, verbose=1),

    # Reduce learning rate when validation loss stops improving
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=5, min_lr=1e-6, verbose=1),

    # Stop training early if no improvement after 10 epochs
    EarlyStopping(monitor='val_loss', patience=10,
                  restore_best_weights=True, verbose=1)
]

# ── 6. TRAIN ──────────────────────────────────────────────
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=test_gen,
    callbacks=callbacks
)

# ── 7. PLOT RESULTS ───────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'],     label='Train acc')
ax1.plot(history.history['val_accuracy'], label='Val acc')
ax1.set_title('Accuracy over epochs')
ax1.legend()

ax2.plot(history.history['loss'],     label='Train loss')
ax2.plot(history.history['val_loss'], label='Val loss')
ax2.set_title('Loss over epochs')
ax2.legend()

plt.savefig('training_plot.png')
plt.show()
print("Training complete. Model saved as emotion_model.h5")
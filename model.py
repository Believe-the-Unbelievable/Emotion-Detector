# model.py
# This file defines our CNN architecture from scratch.
# No pretrained weights — every layer is built fresh.

import tensorflow as tf
from tensorflow.keras import layers, models

def build_emotion_model(num_classes=7):
    """
    A custom CNN built from scratch for emotion classification.
    Input: 48x48 grayscale images
    Output: probability scores for 7 emotion classes
    """
    model = models.Sequential([

        # --- BLOCK 1 ---
        # Conv layer: 32 filters, 3x3 kernel, 'same' padding keeps spatial dims intact
        layers.Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)),
        layers.BatchNormalization(),   # stabilizes training
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),     # halves spatial dims: 48x48 → 24x24
        layers.Dropout(0.25),          # prevents overfitting

        # --- BLOCK 2 ---
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),     # 24x24 → 12x12
        layers.Dropout(0.25),

        # --- BLOCK 3 ---
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),     # 12x12 → 6x6
        layers.Dropout(0.25),

        # --- FULLY CONNECTED HEAD ---
        layers.Flatten(),              # 6x6x128 = 4608 values → 1D vector
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        # Output layer: 7 neurons, softmax gives probabilities
        layers.Dense(num_classes, activation='softmax')
    ])

    return model
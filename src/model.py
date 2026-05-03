"""Reusable CNN architecture for concrete crack binary classification."""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D


def build_crack_cnn(input_shape=(200, 200, 3)):
    """Build the same CNN topology used in the notebook in a reusable form."""
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2))

    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((2, 2), strides=2))

    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((2, 2), strides=2))

    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((2, 2), strides=2))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    return model

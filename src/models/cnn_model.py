"""
1D CNN model for ECG arrhythmia classification.
"""

from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from .contract import CLASS_NAMES, PREPROCESSING, PROTOCOL


def build_ecg_cnn(input_shape: Tuple[int, int] = (187, 1), num_classes: int = 5) -> keras.Model:
    """
    Build 1D CNN for ECG classification.

    Architecture:
    - 3 Conv1D blocks with batch norm and max pooling
    - Global average pooling
    - Dense layers with dropout
    - Softmax output

    Args:
        input_shape: Shape of input (timesteps, channels)
        num_classes: Number of arrhythmia classes

    Returns:
        Compiled Keras model
    """
    model = models.Sequential(
        [
            # Block 1
            layers.Conv1D(64, kernel_size=6, activation="relu", input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=3),
            layers.Dropout(0.2),
            # Block 2
            layers.Conv1D(128, kernel_size=6, activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=3),
            layers.Dropout(0.2),
            # Block 3
            layers.Conv1D(256, kernel_size=6, activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            # Dense layers
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            # Output
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def build_residual_cnn(
    input_shape: Tuple[int, int] = (187, 1), num_classes: int = 5
) -> keras.Model:
    """
    Build residual CNN with skip connections for better gradient flow.

    Args:
        input_shape: Shape of input (timesteps, channels)
        num_classes: Number of arrhythmia classes

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)

    # Initial conv
    x = layers.Conv1D(64, kernel_size=7, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Residual block 1
    shortcut = x
    x = layers.Conv1D(64, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(64, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Residual block 2
    shortcut = layers.Conv1D(128, kernel_size=1, padding="same")(x)
    x = layers.Conv1D(128, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(128, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Global pooling and output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


@dataclass(frozen=True)
class ArrhythmiaClassifier:
    """An explicitly supplied model and its metadata; no implicit random-weight state."""

    model: keras.Model
    metadata: dict
    CLASS_NAMES = CLASS_NAMES

    @classmethod
    def load_model(cls, path: Path):
        metadata = json.loads(path.with_suffix(".json").read_text())
        assert metadata["protocol"] == PROTOCOL, "Model label protocol is incompatible"
        assert metadata["preprocessing"] == PREPROCESSING, "Model preprocessing is incompatible"
        assert metadata["class_names"] == CLASS_NAMES, "Model class order is incompatible"
        assert (
            metadata["model_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        ), "Model checksum mismatch"
        assert (
            metadata["trained"] is True and metadata["data_kind"] == "public"
        ), "Artifact is synthetic or untrained"
        model = keras.models.load_model(path, compile=False)
        assert model.input_shape[1:] == (187, 1), "Expected 187-sample beats"
        assert model.output_shape[-1] == len(CLASS_NAMES), "Model class count is incompatible"
        return cls(model, metadata)

    def predict(self, ecg_segment: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        assert ecg_segment.shape == (187,), "Expected exactly one 187-sample beat"
        assert np.all(np.isfinite(ecg_segment)), "Segment must be finite"
        probabilities = self.model.predict(ecg_segment.reshape(1, 187, 1), verbose=0)[0]
        index = int(np.argmax(probabilities))
        return (
            CLASS_NAMES[index],
            float(probabilities[index]),
            dict(zip(CLASS_NAMES, map(float, probabilities))),
        )

    def predict_batch(self, ecg_segments: np.ndarray) -> np.ndarray:
        return self.model.predict(ecg_segments, verbose=0)

"""
1D CNN model for ECG arrhythmia classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from typing import Tuple, Dict
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


class ArrhythmiaClassifier:
    """
    Wrapper class for arrhythmia classification.
    """

    CLASS_NAMES = CLASS_NAMES

    def __init__(self, model_path: str = None, allow_untrained: bool = False):
        """
        Initialize classifier.

        Args:
            model_path: Path to saved model weights
        """
        self.model = None
        self.model_path = model_path
        self.allow_untrained = allow_untrained
        self.metadata = None
        self.ready = False

        if model_path:
            self.load_model(model_path)
        elif allow_untrained:
            self.model = build_ecg_cnn()

    def load_model(self, model_path: str):
        """Load model from file."""
        path = Path(model_path)
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata.get("protocol") != PROTOCOL or metadata.get("preprocessing") != PREPROCESSING:
            raise ValueError("Model preprocessing/label protocol is incompatible")
        if metadata.get("class_names") != CLASS_NAMES:
            raise ValueError("Model class order is incompatible")
        if metadata.get("model_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Model checksum mismatch")
        ready = metadata.get("trained") is True and metadata.get("data_kind") == "public"
        if not ready and not self.allow_untrained:
            raise ValueError("Artifact is synthetic or untrained")
        model = keras.models.load_model(path, compile=False)
        if model.input_shape[1:] != (187, 1) or model.output_shape[-1] != len(CLASS_NAMES):
            raise ValueError("Model dimensions do not match its declared contract")
        self.model = model
        self.metadata = metadata
        self.ready = ready

    def save_model(self, model_path: str):
        """Save model to file."""
        if self.model is None:
            raise RuntimeError("No model to save")
        self.model.save(model_path)
        path = Path(model_path)
        metadata = {
            "protocol": PROTOCOL,
            "preprocessing": PREPROCESSING,
            "class_names": CLASS_NAMES,
            "trained": False,
            "data_kind": "synthetic",
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))

    def predict(self, ecg_segment: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict arrhythmia class for ECG segment.

        Args:
            ecg_segment: Preprocessed ECG segment (187 samples)

        Returns:
            Tuple of (predicted class, confidence, all probabilities)
        """
        if self.model is None or not (self.ready or self.allow_untrained):
            raise RuntimeError("No compatible trained model; classification unavailable")
        ecg_segment = np.asarray(ecg_segment)
        if not np.all(np.isfinite(ecg_segment)):
            raise ValueError("Segment must be finite")
        if len(ecg_segment.shape) == 1:
            ecg_segment = ecg_segment.reshape(1, -1, 1)

        if ecg_segment.shape != (1, 187, 1):
            raise ValueError("Expected exactly one 187-sample beat")
        predictions = self.model.predict(ecg_segment, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]

        probabilities = {name: float(prob) for name, prob in zip(self.CLASS_NAMES, predictions)}

        return self.CLASS_NAMES[predicted_class_idx], float(confidence), probabilities

    def predict_batch(self, ecg_segments: np.ndarray) -> np.ndarray:
        """
        Predict multiple ECG segments.

        Args:
            ecg_segments: Array of ECG segments (N, 187, 1)

        Returns:
            Array of predictions
        """
        if self.model is None or not (self.ready or self.allow_untrained):
            raise RuntimeError("No compatible trained model")
        return self.model.predict(ecg_segments, verbose=0)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            X_train: Training data
            y_train: Training labels (one-hot encoded)
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
        )

        return history

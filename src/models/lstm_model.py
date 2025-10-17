"""
LSTM/RNN model for temporal ECG pattern recognition.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from typing import Tuple


def build_lstm_model(
    input_shape: Tuple[int, int] = (187, 1),
    num_classes: int = 5
) -> keras.Model:
    """
    Build bidirectional LSTM for ECG classification.

    Args:
        input_shape: Shape of input (timesteps, channels)
        num_classes: Number of arrhythmia classes

    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Bidirectional(
            layers.LSTM(64, return_sequences=True),
            input_shape=input_shape
        ),
        layers.Dropout(0.3),

        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_cnn_lstm_hybrid(
    input_shape: Tuple[int, int] = (187, 1),
    num_classes: int = 5
) -> keras.Model:
    """
    Build hybrid CNN-LSTM model combining spatial and temporal features.

    Args:
        input_shape: Shape of input (timesteps, channels)
        num_classes: Number of arrhythmia classes

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)

    # CNN feature extraction
    x = layers.Conv1D(64, kernel_size=5, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=5, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # LSTM temporal modeling
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.3)(x)

    # Dense output
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_attention_lstm(
    input_shape: Tuple[int, int] = (187, 1),
    num_classes: int = 5
) -> keras.Model:
    """
    Build LSTM with attention mechanism.

    Args:
        input_shape: Shape of input (timesteps, channels)
        num_classes: Number of arrhythmia classes

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)

    # LSTM layers
    lstm_out = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True)
    )(inputs)

    # Attention mechanism
    attention = layers.Dense(1, activation='tanh')(lstm_out)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(128)(attention)
    attention = layers.Permute([2, 1])(attention)

    # Apply attention
    sent_representation = layers.Multiply()([lstm_out, attention])
    sent_representation = layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=1)
    )(sent_representation)

    # Output
    x = layers.Dense(64, activation='relu')(sent_representation)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

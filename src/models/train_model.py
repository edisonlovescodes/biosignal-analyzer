"""
Training pipeline for ECG arrhythmia classification models.
"""

import numpy as np
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import os
from typing import Tuple, List
from .cnn_model import build_ecg_cnn, build_residual_cnn


# MIT-BIH Arrhythmia Database annotation mapping
ANNOTATION_MAP = {
    'N': 'Normal',           # Normal beat
    'L': 'Normal',           # Left bundle branch block
    'R': 'Normal',           # Right bundle branch block
    'A': 'PAC',              # Atrial premature beat
    'a': 'PAC',              # Aberrated atrial premature beat
    'J': 'PAC',              # Nodal (junctional) premature beat
    'S': 'PAC',              # Supraventricular premature beat
    'V': 'PVC',              # Premature ventricular contraction
    'F': 'Atrial Fibrillation',  # Fusion of ventricular and normal
    'f': 'Atrial Fibrillation',  # Fusion of paced and normal
    '/': 'Other',            # Paced beat
    'Q': 'Other',            # Unclassifiable beat
    '?': 'Other'             # Beat not classified during learning
}


def load_mitbih_record(record_path: str, segment_length: int = 187) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load and segment a single MIT-BIH record.

    Args:
        record_path: Path to record (without extension)
        segment_length: Length of each segment

    Returns:
        Tuple of (segments, labels)
    """
    try:
        # Load signal
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0]  # Use first channel

        # Load annotations
        annotation = wfdb.rdann(record_path, 'atr')
        beat_locations = annotation.sample
        beat_types = annotation.symbol

        segments = []
        labels = []

        half_length = segment_length // 2

        for loc, beat_type in zip(beat_locations, beat_types):
            # Map annotation to our classes
            if beat_type not in ANNOTATION_MAP:
                continue

            label = ANNOTATION_MAP[beat_type]

            # Extract segment around beat
            start = max(0, loc - half_length)
            end = min(len(signal), loc + half_length)

            if end - start == segment_length:
                segment = signal[start:end]
                segments.append(segment)
                labels.append(label)

        return segments, labels

    except Exception as e:
        print(f"Error loading record {record_path}: {e}")
        return [], []


def load_mitbih_dataset(data_dir: str, records: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MIT-BIH dataset from multiple records.

    Args:
        data_dir: Directory containing MIT-BIH records
        records: List of record names (e.g., ['100', '101', ...])

    Returns:
        Tuple of (X, y) where X is segments and y is labels
    """
    if records is None:
        # Default MIT-BIH records for training
        records = ['100', '101', '103', '105', '106', '108', '109', '111',
                   '112', '113', '114', '115', '116', '117', '118', '119',
                   '121', '122', '123', '124', '200', '201', '202', '203',
                   '205', '207', '208', '209', '210', '212', '213', '214',
                   '215', '217', '219', '220', '221', '222', '223', '228',
                   '230', '231', '232', '233', '234']

    all_segments = []
    all_labels = []

    for record in records:
        record_path = os.path.join(data_dir, record)
        segments, labels = load_mitbih_record(record_path)
        all_segments.extend(segments)
        all_labels.extend(labels)

    X = np.array(all_segments)
    y = np.array(all_labels)

    return X, y


def prepare_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple:
    """
    Prepare data for training.

    Args:
        X: Feature array
        y: Labels
        test_size: Fraction for validation

    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # Normalize
    X_normalized = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)

    # Reshape for CNN
    X_reshaped = X_normalized.reshape(X_normalized.shape[0], X_normalized.shape[1], 1)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_reshaped, y_categorical,
        test_size=test_size,
        stratify=y_categorical.argmax(axis=1),
        random_state=42
    )

    return X_train, X_val, y_train, y_val, label_encoder


def train_ecg_model(
    data_dir: str,
    save_path: str = 'data/models/ecg_model.h5',
    model_type: str = 'cnn',
    epochs: int = 50
):
    """
    Main training function.

    Args:
        data_dir: Directory with MIT-BIH data
        save_path: Where to save trained model
        model_type: 'cnn' or 'residual'
        epochs: Training epochs
    """
    print("Loading MIT-BIH dataset...")
    X, y = load_mitbih_dataset(data_dir)

    print(f"Loaded {len(X)} samples")
    print("Preparing data...")
    X_train, X_val, y_train, y_val, label_encoder = prepare_data(X, y)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Classes: {label_encoder.classes_}")

    print("Building model...")
    if model_type == 'cnn':
        model = build_ecg_cnn(input_shape=(187, 1), num_classes=len(label_encoder.classes_))
    else:
        model = build_residual_cnn(input_shape=(187, 1), num_classes=len(label_encoder.classes_))

    print("Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )

    print(f"Saving model to {save_path}...")
    model.save(save_path)

    # Also save label encoder
    import pickle
    with open(save_path.replace('.h5', '_labels.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    print("Training complete!")
    return model, history, label_encoder


if __name__ == "__main__":
    # Example usage
    train_ecg_model(
        data_dir="data/mit-bih-arrhythmia-database-1.0.0",
        save_path="data/models/ecg_arrhythmia_classifier.h5"
    )

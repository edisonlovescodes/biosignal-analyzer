"""
Quick training script using synthetic data.
Creates a demo model in ~2 minutes for testing purposes.

NOTE: This is NOT as accurate as training on real MIT-BIH data,
but it allows you to test the full functionality of the app.

Run: python quick_train_demo.py
"""

import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.cnn_model import build_ecg_cnn


def generate_synthetic_heartbeats(n_samples=5000, segment_length=187):
    """Generate synthetic ECG heartbeats for demo purposes."""
    print("🔬 Generating synthetic training data...")

    X = []
    y = []

    classes = ['Normal', 'Atrial Fibrillation', 'PVC', 'PAC', 'Other']
    samples_per_class = n_samples // len(classes)

    for cls_idx, cls_name in enumerate(classes):
        for _ in range(samples_per_class):
            # Generate synthetic heartbeat
            t = np.linspace(0, 1, segment_length)

            if cls_name == 'Normal':
                # Normal sinus rhythm
                signal = np.exp(-((t - 0.3) ** 2) / 0.01) * 1.5  # R-peak
                signal += np.exp(-((t - 0.6) ** 2) / 0.02) * 0.3  # T-wave
                signal -= 0.1 * np.sin(2 * np.pi * 0.5 * t)  # Baseline

            elif cls_name == 'PVC':
                # Premature ventricular contraction (wider QRS)
                signal = np.exp(-((t - 0.35) ** 2) / 0.03) * 2.0  # Wide R-peak
                signal += np.exp(-((t - 0.65) ** 2) / 0.02) * 0.4  # T-wave

            elif cls_name == 'PAC':
                # Premature atrial contraction
                signal = np.exp(-((t - 0.25) ** 2) / 0.01) * 1.3  # Early beat
                signal += np.exp(-((t - 0.55) ** 2) / 0.02) * 0.3

            elif cls_name == 'Atrial Fibrillation':
                # Irregular rhythm
                signal = np.exp(-((t - 0.3) ** 2) / 0.015) * 1.4
                signal += 0.2 * np.random.randn(segment_length)  # More noise

            else:  # Other
                # Artifact or unusual pattern
                signal = 0.5 * np.sin(10 * np.pi * t)
                signal += 0.3 * np.random.randn(segment_length)

            # Add realistic noise
            noise = np.random.randn(segment_length) * 0.05
            signal += noise

            X.append(signal)
            y.append(cls_name)

    return np.array(X), np.array(y)


def train_demo_model():
    """Train a demo model on synthetic data."""
    print("=" * 60)
    print("🫀 Quick Demo Model Training")
    print("=" * 60)
    print("\n⚠️  Note: This uses synthetic data for demo purposes.")
    print("   For real accuracy, use download_and_train.py with MIT-BIH data.\n")

    # Generate data
    X, y = generate_synthetic_heartbeats(n_samples=5000)
    print(f"✅ Generated {len(X):,} synthetic heartbeats\n")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # Normalize
    X_normalized = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    X_reshaped = X_normalized.reshape(-1, 187, 1)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_reshaped, y_categorical,
        test_size=0.2,
        random_state=42
    )

    print(f"📊 Training samples: {len(X_train):,}")
    print(f"📊 Validation samples: {len(X_val):,}")
    print(f"📊 Classes: {list(label_encoder.classes_)}\n")

    # Build model
    print("🏗️  Building model...")
    model = build_ecg_cnn(input_shape=(187, 1), num_classes=5)

    print("🚀 Training (this takes ~2 minutes)...\n")

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=1
    )

    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    print(f"\n📊 Results:")
    print(f"   Training Accuracy: {train_acc*100:.2f}%")
    print(f"   Validation Accuracy: {val_acc*100:.2f}%")

    # Save
    os.makedirs('data/models', exist_ok=True)
    model.save('data/models/ecg_arrhythmia_classifier.h5')

    import pickle
    with open('data/models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    print("\n✅ Demo model saved!")
    print("   Restart your Streamlit app to use it.\n")
    print("=" * 60)
    print("🎉 Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    train_demo_model()

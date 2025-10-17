"""
Download MIT-BIH dataset and train the arrhythmia classification model.

This script will:
1. Download the MIT-BIH Arrhythmia Database (~650MB)
2. Process the data into training samples
3. Train a CNN model (~45 minutes on CPU, ~5 minutes on GPU)
4. Save the trained model for use in the app

Run: python download_and_train.py
"""

import os
import sys
import wfdb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.cnn_model import build_ecg_cnn
from src.signal_processing.filters import normalize_signal


# MIT-BIH annotation mapping
ANNOTATION_MAP = {
    'N': 'Normal',
    'L': 'Normal',
    'R': 'Normal',
    'A': 'PAC',
    'a': 'PAC',
    'J': 'PAC',
    'S': 'PAC',
    'V': 'PVC',
    'F': 'Atrial Fibrillation',
    'f': 'Atrial Fibrillation',
    '/': 'Other',
    'Q': 'Other',
}


def download_mitbih_database(data_dir='data/mit-bih'):
    """Download MIT-BIH Arrhythmia Database."""
    print("📥 Downloading MIT-BIH Arrhythmia Database...")
    print("   This may take 5-10 minutes (~650MB download)")

    os.makedirs(data_dir, exist_ok=True)

    try:
        wfdb.dl_database('mitdb', data_dir)
        print("✅ Download complete!")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def load_record(record_path, segment_length=187):
    """Load and segment a single MIT-BIH record."""
    try:
        # Load signal
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0]

        # Load annotations
        annotation = wfdb.rdann(record_path, 'atr')
        beat_locations = annotation.sample
        beat_types = annotation.symbol

        segments = []
        labels = []

        half_length = segment_length // 2

        for loc, beat_type in zip(beat_locations, beat_types):
            if beat_type not in ANNOTATION_MAP:
                continue

            label = ANNOTATION_MAP[beat_type]

            # Extract segment
            start = max(0, loc - half_length)
            end = min(len(signal), loc + half_length)

            if end - start == segment_length:
                segment = signal[start:end]
                segments.append(segment)
                labels.append(label)

        return segments, labels

    except Exception as e:
        print(f"⚠️  Error loading record {record_path}: {e}")
        return [], []


def prepare_dataset(data_dir='data/mit-bih'):
    """Prepare complete dataset."""
    print("\n📊 Processing MIT-BIH records...")

    # MIT-BIH record numbers
    records = ['100', '101', '103', '105', '106', '108', '109', '111',
               '112', '113', '114', '115', '116', '117', '118', '119',
               '121', '122', '123', '124', '200', '201', '202', '203',
               '205', '207', '208', '209', '210', '212', '213', '214',
               '215', '217', '219', '220', '221', '222', '223', '228',
               '230', '231', '232', '233', '234']

    all_segments = []
    all_labels = []

    for i, record in enumerate(records):
        print(f"   Processing record {record} ({i+1}/{len(records)})...", end='\r')
        record_path = os.path.join(data_dir, record)
        segments, labels = load_record(record_path)
        all_segments.extend(segments)
        all_labels.extend(labels)

    print(f"\n✅ Processed {len(all_segments)} heartbeats!")

    # Convert to arrays
    X = np.array(all_segments)
    y = np.array(all_labels)

    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\n📈 Dataset distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"   {cls}: {cnt:,} beats ({cnt/len(y)*100:.1f}%)")

    return X, y


def train_model(X, y, epochs=50):
    """Train the CNN model."""
    print("\n🤖 Preparing training data...")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # Normalize
    X_normalized = np.array([normalize_signal(x) for x in X])

    # Reshape for CNN
    X_reshaped = X_normalized.reshape(X_normalized.shape[0], X_normalized.shape[1], 1)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_reshaped, y_categorical,
        test_size=0.2,
        stratify=y_categorical.argmax(axis=1),
        random_state=42
    )

    print(f"   Training samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,}")
    print(f"   Classes: {list(label_encoder.classes_)}")

    # Build model
    print("\n🏗️  Building CNN model...")
    model = build_ecg_cnn(input_shape=(187, 1), num_classes=len(label_encoder.classes_))

    print(f"\n🚀 Starting training ({epochs} epochs)...")
    print("   This will take ~45 minutes on CPU, ~5 minutes on GPU")
    print("   You can watch the progress below:\n")

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
    )

    # Evaluate
    print("\n📊 Final Results:")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    print(f"   Training Accuracy: {train_acc*100:.2f}%")
    print(f"   Validation Accuracy: {val_acc*100:.2f}%")

    return model, label_encoder, history


def save_model(model, label_encoder, save_dir='data/models'):
    """Save trained model and label encoder."""
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, 'ecg_arrhythmia_classifier.h5')
    labels_path = os.path.join(save_dir, 'label_encoder.pkl')

    print(f"\n💾 Saving model to {model_path}...")
    model.save(model_path)

    print(f"💾 Saving label encoder to {labels_path}...")
    import pickle
    with open(labels_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    print("\n✅ Model saved successfully!")
    print(f"\n🎉 Training complete! Your app now has a trained arrhythmia classifier.")
    print(f"   Restart your Streamlit app to use the trained model.")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("🫀 BioSignal Analyzer - Model Training")
    print("=" * 60)

    # Step 1: Download data
    if not os.path.exists('data/mit-bih/100.dat'):
        success = download_mitbih_database()
        if not success:
            print("\n❌ Failed to download dataset. Please try again.")
            return
    else:
        print("✅ MIT-BIH database already downloaded")

    # Step 2: Prepare dataset
    X, y = prepare_dataset()

    if len(X) == 0:
        print("❌ No data loaded. Cannot train model.")
        return

    # Step 3: Train model
    import tensorflow as tf
    model, label_encoder, history = train_model(X, y, epochs=50)

    # Step 4: Save model
    save_model(model, label_encoder)

    print("\n" + "=" * 60)
    print("🎊 All done! Your BioSignal Analyzer is now fully trained!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Local public-data training; split subjects before pooling beats. No automatic downloads."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import wfdb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.utils import to_categorical
from .cnn_model import build_ecg_cnn, build_residual_cnn
from .contract import (
    ANNOTATION_MAP,
    CLASS_NAMES,
    PREPROCESSING,
    PROTOCOL,
    extract_beat,
    filter_record,
    subject_id,
)


def load_mitbih_record(record_path, segment_length=187):
    record = wfdb.rdrecord(str(record_path))
    values = filter_record(record.p_signal[:, 0], record.fs)
    annotation = wfdb.rdann(str(record_path), "atr")
    segments = []
    labels = []
    for loc, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in ANNOTATION_MAP:
            continue
        segment = extract_beat(values, loc, segment_length)
        if segment is not None:
            segments.append(segment)
            labels.append(ANNOTATION_MAP[symbol])
    return segments, labels


def load_mitbih_dataset(data_dir, records=None, return_groups=False):
    records = records or sorted(p.stem for p in Path(data_dir).glob("*.hea"))
    if not records:
        raise ValueError("No local WFDB records found")
    segments = []
    labels = []
    groups = []
    for record in records:
        x, y = load_mitbih_record(Path(data_dir) / record)
        segments.extend(x)
        labels.extend(y)
        groups.extend([subject_id(record)] * len(x))
    if not segments:
        raise ValueError("No complete annotated beats extracted")
    result = (np.asarray(segments), np.asarray(labels))
    return (*result, np.asarray(groups)) if return_groups else result


def split_subjects(groups, seed=42):
    if len(set(groups)) < 5:
        raise ValueError("Need at least five subjects for train/validation/test")
    indices = np.arange(len(groups))
    train_val, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            indices, groups=groups
        )
    )
    tr, val = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
            train_val, groups=groups[train_val]
        )
    )
    return {"train": train_val[tr], "validation": train_val[val], "test": test}


def train_ecg_model(
    data_dir, save_path="data/models/ecg_arrhythmia_classifier.keras", model_type="cnn", epochs=10
):
    if model_type not in {"cnn", "residual"} or not 1 <= epochs <= 100:
        raise ValueError("Invalid training configuration")
    tf.keras.utils.set_random_seed(42)
    X, y, groups = load_mitbih_dataset(data_dir, return_groups=True)
    splits = split_subjects(groups)
    labels = np.array([CLASS_NAMES.index(label) for label in y])
    X = X[..., None]
    onehot = to_categorical(labels, num_classes=len(CLASS_NAMES))
    build = build_ecg_cnn if model_type == "cnn" else build_residual_cnn
    model = build(input_shape=(187, 1), num_classes=len(CLASS_NAMES))
    tr = splits["train"]
    val = splits["validation"]
    test = splits["test"]
    history = model.fit(
        X[tr],
        onehot[tr],
        validation_data=(X[val], onehot[val]),
        epochs=epochs,
        batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )
    predicted = model.predict(X[test], verbose=0).argmax(axis=1)
    report = classification_report(
        labels[test],
        predicted,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    metadata = {
        "protocol": PROTOCOL,
        "preprocessing": PREPROCESSING,
        "class_names": CLASS_NAMES,
        "trained": True,
        "data_kind": "public",
        "seed": 42,
        "epochs_requested": epochs,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "split_subjects": {k: sorted(set(groups[idx])) for k, idx in splits.items()},
        "split_beats": {k: len(idx) for k, idx in splits.items()},
        "test_report": report,
        "test_confusion_matrix": confusion_matrix(
            labels[test], predicted, labels=list(range(len(CLASS_NAMES)))
        ).tolist(),
        "limitations": [
            "Custom coarse beat taxonomy; not atrial-fibrillation rhythm detection.",
            "Peak detector and classifier require separate evaluation; no clinical validation.",
        ],
    }
    metadata["record_checksums"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(data_dir).iterdir()
        if p.suffix in {".hea", ".dat", ".atr"}
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return model, history, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", default="data/models/ecg_arrhythmia_classifier.keras")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train_ecg_model(args.data, args.output, epochs=args.epochs)


if __name__ == "__main__":
    main()

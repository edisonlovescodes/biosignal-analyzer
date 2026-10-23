from types import SimpleNamespace
import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.models.contract import ANNOTATION_MAP, CLASS_NAMES, extract_beat, subject_id
from src.models.cnn_model import ArrhythmiaClassifier
from src.models.train_model import load_mitbih_record, split_subjects
from src.signal_processing.filters import notch_filter
from src.signal_processing.features import calculate_hrv_frequency_domain
from api.main import create_app


def test_odd_segments_have_187_samples():
    signal = np.arange(500.0)
    assert extract_beat(signal, 250).shape == (187,)
    assert extract_beat(signal, 20) is None
    assert extract_beat(signal, 450) is None


def test_actual_loader_uses_current_labels_and_segments(monkeypatch):
    import src.models.train_model as training

    monkeypatch.setattr(
        training.wfdb,
        "rdrecord",
        lambda _: SimpleNamespace(p_signal=np.sin(np.arange(1000))[:, None], fs=360),
    )
    monkeypatch.setattr(
        training.wfdb,
        "rdann",
        lambda *a: SimpleNamespace(sample=[200, 400, 600], symbol=["N", "F", "f"]),
    )
    x, y = load_mitbih_record("synthetic")
    assert len(x) == 3
    assert all(len(s) == 187 for s in x)
    assert y == ["Normal/Conduction", "Fusion", "Fusion"]
    assert "Atrial Fibrillation" not in CLASS_NAMES


def test_shared_subject_never_crosses_partitions():
    groups = np.array(
        [subject_id(r) for r in ["100", "101", "103", "105", "106", "201", "202"] for _ in range(3)]
    )
    splits = split_subjects(groups)
    sets = [set(groups[idx]) for idx in splits.values()]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    assert sum("201-202" in s for s in sets) == 1


def test_notch_at_requested_hz_preserves_other_frequency():
    fs = 360
    t = np.arange(3600) / fs
    unwanted = np.sin(2 * np.pi * 60 * t)
    wanted = np.sin(2 * np.pi * 10 * t)
    out = notch_filter(unwanted + wanted, 60, fs)[360:-360]
    middle = t[360:-360]
    amplitude60 = abs(2 * np.mean(out * np.exp(-2j * np.pi * 60 * middle)))
    amplitude10 = abs(2 * np.mean(out * np.exp(-2j * np.pi * 10 * middle)))
    assert amplitude60 < 0.02
    assert amplitude10 > 0.95


def test_hrv_power_uses_frequency_spacing():
    from scipy import signal

    rr = 1000 + 50 * np.sin(np.arange(1000) * 2 * np.pi * 0.1)
    result = calculate_hrv_frequency_domain(rr)
    assert all(np.isfinite(v) and v >= 0 for v in result.values())
    assert 500 < result["lf_power"] < 1600  # ms^2; expected sinusoid variance is about 1250


def test_default_classifier_cannot_predict_random_weights():
    with pytest.raises(TypeError):
        ArrhythmiaClassifier()


def csv(values):
    return ("signal\n" + "\n".join(map(str, values))).encode()


def test_api_without_model_can_explore_but_cannot_classify(tmp_path):
    with TestClient(create_app(tmp_path / "missing.keras")) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert (
            c.post(
                "/predict", files={"file": ("beat.csv", csv(np.ones(187)), "text/csv")}
            ).status_code
            == 503
        )
        r = c.post("/analyze", files={"file": ("flat.csv", csv(np.ones(1000)), "text/csv")})
        assert r.status_code == 200
        assert r.json()["predictions"] == []
        assert r.json()["classification_available"] is False
        assert r.json()["analysis"]["num_peaks"] == 0
        for data in [b"", b"signal\nnope", b"signal\nNaN", csv([1, 2, 3])]:
            assert (
                c.post("/analyze", files={"file": ("bad.csv", data, "text/csv")}).status_code == 400
            )


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol", "unknown"),
        ("class_names", list(reversed(CLASS_NAMES))),
        ("model_sha256", "wrong"),
        ("trained", False),
        ("data_kind", "synthetic"),
    ],
)
def test_incompatible_artifacts_fail_at_startup(tmp_path, field, value):
    import hashlib
    import json
    from src.models.contract import PREPROCESSING, PROTOCOL

    path = tmp_path / "model.keras"
    path.write_bytes(b"fixture, not a trained artifact")
    metadata = {
        "protocol": PROTOCOL,
        "preprocessing": PREPROCESSING,
        "class_names": CLASS_NAMES,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "trained": True,
        "data_kind": "public",
    }
    metadata[field] = value
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(AssertionError):
        with TestClient(create_app(path)):
            pass

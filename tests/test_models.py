"""
Unit tests for ML models.
"""

import pytest
import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn_model import build_ecg_cnn, build_residual_cnn, ArrhythmiaClassifier
from src.models.lstm_model import build_lstm_model, build_cnn_lstm_hybrid


class TestModelArchitectures:
    """Test model building functions."""

    def test_build_ecg_cnn(self):
        model = build_ecg_cnn(input_shape=(187, 1), num_classes=5)

        assert model is not None
        assert len(model.layers) > 0
        assert model.input_shape == (None, 187, 1)
        assert model.output_shape == (None, 5)

    def test_build_residual_cnn(self):
        model = build_residual_cnn(input_shape=(187, 1), num_classes=5)

        assert model is not None
        assert model.input_shape == (None, 187, 1)
        assert model.output_shape == (None, 5)

    def test_build_lstm_model(self):
        model = build_lstm_model(input_shape=(187, 1), num_classes=5)

        assert model is not None
        assert model.input_shape == (None, 187, 1)
        assert model.output_shape == (None, 5)

    def test_build_cnn_lstm_hybrid(self):
        model = build_cnn_lstm_hybrid(input_shape=(187, 1), num_classes=5)

        assert model is not None
        assert model.input_shape == (None, 187, 1)
        assert model.output_shape == (None, 5)


class TestArrhythmiaClassifier:
    """Test ArrhythmiaClassifier class."""

    def test_classifier_initialization(self):
        classifier = ArrhythmiaClassifier()

        assert classifier.model is not None
        assert len(classifier.CLASS_NAMES) == 5

    def test_prediction(self):
        classifier = ArrhythmiaClassifier()

        # Create random ECG segment
        ecg_segment = np.random.randn(187)

        pred_class, confidence, probs = classifier.predict(ecg_segment)

        assert pred_class in classifier.CLASS_NAMES
        assert 0 <= confidence <= 1
        assert len(probs) == 5
        assert all(0 <= p <= 1 for p in probs.values())
        assert abs(sum(probs.values()) - 1.0) < 1e-5  # Probabilities sum to 1

    def test_batch_prediction(self):
        classifier = ArrhythmiaClassifier()

        # Create batch of segments
        batch = np.random.randn(10, 187, 1)

        predictions = classifier.predict_batch(batch)

        assert predictions.shape == (10, 5)
        # Each row should sum to ~1 (softmax)
        assert all(abs(row.sum() - 1.0) < 1e-5 for row in predictions)

    def test_prediction_with_2d_input(self):
        classifier = ArrhythmiaClassifier()

        # Input should be auto-reshaped
        ecg_segment = np.random.randn(187)

        pred_class, confidence, probs = classifier.predict(ecg_segment)

        assert pred_class is not None
        assert confidence is not None


class TestModelTraining:
    """Test model training functionality."""

    def test_model_compilation(self):
        model = build_ecg_cnn()

        # Check if model is compiled
        assert model.optimizer is not None
        assert model.loss is not None

    def test_training_step(self):
        classifier = ArrhythmiaClassifier()

        # Create dummy data
        X_train = np.random.randn(50, 187, 1)
        y_train = tf.keras.utils.to_categorical(np.random.randint(0, 5, 50), 5)

        X_val = np.random.randn(10, 187, 1)
        y_val = tf.keras.utils.to_categorical(np.random.randint(0, 5, 10), 5)

        # Train for 1 epoch
        history = classifier.train(X_train, y_train, X_val, y_val, epochs=1, batch_size=10)

        assert history is not None
        assert 'loss' in history.history
        assert 'accuracy' in history.history


class TestModelSaveLoad:
    """Test model saving and loading."""

    def test_save_and_load(self, tmp_path):
        # Create and save model
        classifier = ArrhythmiaClassifier()
        model_path = tmp_path / "test_model.h5"

        classifier.save_model(str(model_path))
        assert model_path.exists()

        # Load model
        loaded_classifier = ArrhythmiaClassifier(model_path=str(model_path))

        # Compare predictions
        ecg_segment = np.random.randn(187)

        pred1, conf1, probs1 = classifier.predict(ecg_segment)
        pred2, conf2, probs2 = loaded_classifier.predict(ecg_segment)

        assert pred1 == pred2
        assert abs(conf1 - conf2) < 1e-5


class TestInputValidation:
    """Test input validation and error handling."""

    def test_wrong_input_shape(self):
        classifier = ArrhythmiaClassifier()

        # Too short
        with pytest.raises((ValueError, Exception)):
            classifier.predict(np.random.randn(50))

    def test_empty_input(self):
        classifier = ArrhythmiaClassifier()

        with pytest.raises((ValueError, Exception)):
            classifier.predict(np.array([]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Historical synthetic-training shortcut retired; it did not establish clinical performance."""

if __name__ == "__main__":
    print("For synthetic architecture/training checks: python -m pytest tests/test_models.py")
    print(
        "For public-data training: python -m src.models.train_model --data /path/to/local/records"
    )

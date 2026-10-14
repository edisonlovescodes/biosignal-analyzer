"""Compatibility entry point for local-data training; no implicit network download.

Use: python download_and_train.py --data /path/to/WFDB/records --epochs 10
Acquire the public dataset separately under its published terms.
"""

from src.models.train_model import main, load_mitbih_record as load_record

if __name__ == "__main__":
    main()

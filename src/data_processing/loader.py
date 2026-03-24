"""
Data loader for the MITM Research Implementation Platform.
Handles Kaggle API download, CSV loading, and train/val/test splitting.
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def download_dataset(kaggle_slug, raw_path="data/raw/"):
    """Download dataset from Kaggle using the Kaggle API.
    
    Works in both local and Google Colab environments.
    
    Args:
        kaggle_slug: Kaggle dataset slug (e.g., 'majedjaber/bot-iot-all-features-5-sample').
        raw_path: Directory to save downloaded files.
    """
    os.makedirs(raw_path, exist_ok=True)
    
    # Check if data already exists
    existing_csvs = glob.glob(os.path.join(raw_path, "*.csv"))
    if existing_csvs:
        print(f"[Loader] Dataset already exists ({len(existing_csvs)} CSV files in {raw_path}). Skipping download.")
        return
    
    print(f"[Loader] Downloading dataset: {kaggle_slug}")
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(kaggle_slug, path=raw_path, unzip=True)
        print(f"[Loader] Dataset downloaded and extracted to {raw_path}")
    except Exception as e:
        print(f"[Loader] Kaggle API download failed: {e}")
        print("[Loader] Trying alternative method (kaggle CLI)...")
        os.system(f"kaggle datasets download -d {kaggle_slug} -p {raw_path} --unzip")
    
    # Verify download
    csvs = glob.glob(os.path.join(raw_path, "**/*.csv"), recursive=True)
    if csvs:
        print(f"[Loader] Found {len(csvs)} CSV files after download.")
    else:
        raise FileNotFoundError(
            f"No CSV files found after download. Please manually download from: "
            f"https://www.kaggle.com/datasets/{kaggle_slug}"
        )


def load_bot_iot(raw_path="data/raw/", sample_frac=None):
    """Load BoT-IoT dataset from CSV files.
    
    Handles multiple CSV files (the 5% sample is split across 4 files).
    
    Args:
        raw_path: Path to directory containing CSV files.
        sample_frac: If set, randomly sample this fraction of data (for quick testing).
    
    Returns:
        pd.DataFrame: Combined dataset.
    """
    csv_files = glob.glob(os.path.join(raw_path, "**/*.csv"), recursive=True)
    
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_path}. "
            f"Run download_dataset() first or place CSV files manually."
        )
    
    print(f"[Loader] Loading {len(csv_files)} CSV file(s)...")
    
    dfs = []
    for f in sorted(csv_files):
        print(f"  Loading: {os.path.basename(f)}")
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    print(f"[Loader] Total records: {len(data):,}")
    print(f"[Loader] Columns ({len(data.columns)}): {list(data.columns[:10])}...")
    
    if sample_frac is not None and sample_frac < 1.0:
        data = data.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"[Loader] Sampled to {len(data):,} records ({sample_frac*100:.0f}%)")
    
    return data


def split_data(X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split data into train/validation/test sets with stratification.
    
    Uses the 70/15/15 split from Paper 1.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        train_ratio: Fraction for training (default: 0.70).
        val_ratio: Fraction for validation (default: 0.15).
        test_ratio: Fraction for testing (default: 0.15).
        seed: Random seed for reproducibility.
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), random_state=seed, stratify=y
    )
    
    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_ratio, random_state=seed, stratify=y_temp
    )
    
    print(f"[Loader] Split: Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")
    print(f"[Loader] Class distribution (train): {dict(pd.Series(y_train).value_counts())}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

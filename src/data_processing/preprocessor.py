"""
Data preprocessor for the MITM Research Implementation Platform.
Implements Paper 1's preprocessing pipeline: cleaning, encoding, scaling, SMOTE, PCA.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


def clean(df):
    """Clean the dataset: handle missing values, infinities, and duplicates.
    
    Follows Paper 1: "missing values were addressed through automatic imputation
    methods while duplicate records were dropped."
    
    Args:
        df: Raw DataFrame.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    initial_size = len(df)
    
    # Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop duplicates
    df = df.drop_duplicates()
    duplicates_dropped = initial_size - len(df)
    
    # Fill missing values with column mean (numeric) or mode (categorical)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")
    
    remaining_nulls = df.isnull().sum().sum()
    print(f"[Preprocessor] Cleaned: {duplicates_dropped:,} duplicates removed, "
          f"{remaining_nulls} remaining nulls, {len(df):,} records remaining.")
    
    return df.reset_index(drop=True)


def identify_columns(df, target_col="attack", label_col="category"):
    """Identify numeric, categorical, and target columns.
    
    Args:
        df: DataFrame.
        target_col: Binary target column name.
        label_col: Multi-class label column name.
    
    Returns:
        tuple: (numeric_cols, categorical_cols, target_col, label_col)
    """
    # Columns to exclude from features
    exclude = set()
    if target_col in df.columns:
        exclude.add(target_col)
    if label_col in df.columns:
        exclude.add(label_col)
    # Also exclude any ID-like or non-feature columns
    for col in df.columns:
        if col.lower() in ["pkseqid", "saddr", "daddr", "sport", "dport"]:
            exclude.add(col)
    
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    categorical_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in exclude]
    
    print(f"[Preprocessor] Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
    print(f"[Preprocessor] Target: '{target_col}', Label: '{label_col}'")
    
    return numeric_cols, categorical_cols, target_col, label_col


def encode_categorical(df, categorical_cols):
    """One-hot encode categorical features.
    
    Paper 1: "categorical features including protocol type and TCP flags were
    further encoded using one-hot encoding."
    
    Args:
        df: DataFrame.
        categorical_cols: List of categorical column names.
    
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded columns.
    """
    if not categorical_cols:
        return df
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)
    print(f"[Preprocessor] One-hot encoded {len(categorical_cols)} categorical columns. "
          f"Total columns: {len(df.columns)}")
    
    return df


def encode_target(y):
    """Encode target labels to numeric values.
    
    Args:
        y: Target Series or array.
    
    Returns:
        tuple: (y_encoded as numpy array, LabelEncoder instance)
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"[Preprocessor] Target classes: {dict(zip(le.classes_, range(len(le.classes_))))}")
    return y_encoded, le


def scale_features(X, method="minmax", scaler=None):
    """Scale features using MinMax or Standard scaling.
    
    Paper 1: "Min-Max Scaling to keep all values standardized within [0, 1]."
    
    Args:
        X: Feature matrix (numpy array or DataFrame).
        method: 'minmax' or 'standard'.
        scaler: Pre-fitted scaler (for val/test sets). If None, fits a new one.
    
    Returns:
        tuple: (X_scaled as numpy array, fitted scaler)
    """
    if scaler is None:
        if method == "minmax":
            scaler = MinMaxScaler(feature_range=(0, 1))
        elif method == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        X_scaled = scaler.fit_transform(X)
        print(f"[Preprocessor] Scaled features using {method}. Shape: {X_scaled.shape}")
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler


def apply_smote(X, y, random_state=42):
    """Apply SMOTE oversampling + random undersampling for class balance.
    
    Paper 1: "SMOTE was used for oversampling the minority class and by using
    random undersampling the impact of the majority class was controlled."
    
    Args:
        X: Feature matrix.
        y: Target vector.
        random_state: Random seed.
    
    Returns:
        tuple: (X_balanced, y_balanced)
    """
    print(f"[Preprocessor] Before SMOTE - Class distribution: {dict(pd.Series(y).value_counts())}")
    
    # SMOTE + Random Undersampling pipeline
    smote = SMOTE(random_state=random_state)
    under = RandomUnderSampler(random_state=random_state)
    pipeline = ImbPipeline([("smote", smote), ("undersample", under)])
    
    X_balanced, y_balanced = pipeline.fit_resample(X, y)
    
    print(f"[Preprocessor] After SMOTE - Class distribution: {dict(pd.Series(y_balanced).value_counts())}")
    print(f"[Preprocessor] Balanced shape: {X_balanced.shape}")
    
    return X_balanced, y_balanced


def apply_pca(X, n_components, pca=None):
    """Apply PCA for dimensionality reduction.
    
    Paper 1: "feature selection using PCA which reduces the amount of
    non-relevant information."
    
    Args:
        X: Feature matrix.
        n_components: Number of principal components to keep.
        pca: Pre-fitted PCA (for val/test sets). If None, fits a new one.
    
    Returns:
        tuple: (X_reduced, fitted PCA)
    """
    if pca is None:
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(X)
        explained = sum(pca.explained_variance_ratio_) * 100
        print(f"[Preprocessor] PCA: {X.shape[1]} → {n_components} features "
              f"({explained:.1f}% variance explained)")
    else:
        X_reduced = pca.transform(X)
    
    return X_reduced, pca


def reshape_for_dl(X, timesteps=1):
    """Reshape 2D feature matrix to 3D for deep learning models.
    
    LSTM/CNN expect input shape (samples, timesteps, features).
    
    Args:
        X: 2D numpy array (samples, features).
        timesteps: Number of timesteps (default: 1 for single-step).
    
    Returns:
        numpy array: 3D array (samples, timesteps, features).
    """
    if len(X.shape) == 2:
        X = X.reshape((X.shape[0], timesteps, X.shape[1]))
    return X


def preprocess_pipeline(df, config):
    """Run the full preprocessing pipeline.
    
    Args:
        df: Raw DataFrame.
        config: Config dict with preprocessing and dataset settings.
    
    Returns:
        dict: {
            'X_train', 'X_val', 'X_test',
            'y_train', 'y_val', 'y_test',
            'scaler', 'label_encoder', 'pca',
            'feature_names'
        }
    """
    from src.data_processing.loader import split_data
    
    dataset_cfg = config["dataset"]
    preproc_cfg = config["preprocessing"]
    
    target_col = dataset_cfg.get("target_column", "attack")
    label_col = dataset_cfg.get("label_column", "category")
    
    # Step 1: Clean
    df = clean(df)
    
    # Step 2: Identify column types
    numeric_cols, categorical_cols, target_col, label_col = identify_columns(
        df, target_col, label_col
    )
    
    # Step 3: Encode categorical features
    df = encode_categorical(df, categorical_cols)
    
    # Step 4: Separate features and target
    # Re-identify numeric columns after encoding
    exclude = {target_col, label_col} & set(df.columns)
    feature_cols = [c for c in df.columns if c not in exclude]
    
    X = df[feature_cols].values.astype(np.float32)
    
    # Encode target (binary: attack or not)
    if target_col in df.columns:
        y, label_encoder = encode_target(df[target_col])
    else:
        raise KeyError(f"Target column '{target_col}' not found in dataset. "
                       f"Available columns: {list(df.columns)}")
    
    # Step 5: Split data (70/15/15)
    split_cfg = dataset_cfg.get("split", {})
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        train_ratio=split_cfg.get("train", 0.70),
        val_ratio=split_cfg.get("val", 0.15),
        test_ratio=split_cfg.get("test", 0.15),
        seed=dataset_cfg.get("random_seed", 42)
    )
    
    # Step 6: Scale features (fit on train only)
    X_train, scaler = scale_features(X_train, method=preproc_cfg.get("scaling_method", "minmax"))
    X_val, _ = scale_features(X_val, scaler=scaler)
    X_test, _ = scale_features(X_test, scaler=scaler)
    
    # Step 7: SMOTE (on train set only)
    pca_model = None
    if preproc_cfg.get("smote", True):
        X_train, y_train = apply_smote(X_train, y_train, random_state=dataset_cfg.get("random_seed", 42))
    
    # Step 8: PCA (optional)
    n_components = preproc_cfg.get("pca_components")
    if n_components is not None:
        X_train, pca_model = apply_pca(X_train, n_components)
        X_val, _ = apply_pca(X_val, n_components, pca=pca_model)
        X_test, _ = apply_pca(X_test, n_components, pca=pca_model)
    
    print(f"\n[Preprocessor] Pipeline complete!")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
    
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler, "label_encoder": label_encoder, "pca": pca_model,
        "feature_names": feature_cols
    }

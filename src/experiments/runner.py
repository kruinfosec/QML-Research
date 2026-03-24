"""
Experiment runner for the MITM Research Implementation Platform.
Orchestrates the full pipeline: data loading, preprocessing, training, evaluation.
"""
import os
import time
import json
import numpy as np
import pandas as pd

from src.config import load_config, get_model_config
from src.data_processing.loader import download_dataset, load_bot_iot, split_data
from src.data_processing.preprocessor import preprocess_pipeline, reshape_for_dl
from src.models.lstm_cnn import LSTMCNNDetector
from src.models.baselines import create_baseline
from src.experiments.metrics import compute_metrics, compare_models, print_results
from src.experiments.visualization import (
    plot_confusion_matrix, plot_roc_curves,
    plot_training_history, plot_comparison_bar
)


def run_all(config_path=None, sample_frac=None):
    """Run the complete experiment pipeline.
    
    Trains the Hybrid LSTM-CNN + all 5 baselines, evaluates, and generates results.
    
    Args:
        config_path: Path to config.yaml (default: project root).
        sample_frac: Sample fraction for quick testing (e.g., 0.1 for 10%).
    
    Returns:
        dict: All results {model_name: metrics_dict}.
    """
    # Load config
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    
    # ===== 1. DATA LOADING =====
    print("\n" + "="*60)
    print("  STEP 1: Data Loading")
    print("="*60)
    
    download_dataset(dataset_cfg["kaggle_slug"], dataset_cfg["raw_path"])
    df = load_bot_iot(dataset_cfg["raw_path"], sample_frac=sample_frac)
    
    # ===== 2. PREPROCESSING =====
    print("\n" + "="*60)
    print("  STEP 2: Preprocessing")
    print("="*60)
    
    data = preprocess_pipeline(df, config)
    
    # Reshape for DL models (samples, 1, features)
    X_train = reshape_for_dl(data["X_train"])
    X_val = reshape_for_dl(data["X_val"])
    X_test = reshape_for_dl(data["X_test"])
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    print(f"\nInput shape for DL models: {input_shape}")
    
    # ===== 3. TRAIN ALL MODELS =====
    all_results = {}
    roc_data = {}
    
    # --- 3a. Hybrid LSTM-CNN (Paper 1 main model) ---
    print("\n" + "="*60)
    print("  STEP 3a: Training Hybrid LSTM-CNN")
    print("="*60)
    
    lstm_cnn_cfg = get_model_config(config, "lstm_cnn")
    lstm_cnn = LSTMCNNDetector(lstm_cnn_cfg)
    lstm_cnn.build(input_shape)
    
    start = time.time()
    lstm_cnn.train(X_train, y_train, X_val, y_val)
    train_time = time.time() - start
    
    metrics = lstm_cnn.evaluate(X_test, y_test)
    metrics["training_time_sec"] = round(train_time, 2)
    all_results["LSTM-CNN (Hybrid)"] = metrics
    print_results("LSTM-CNN (Hybrid)", metrics)
    
    # Save LSTM-CNN training history
    plot_training_history(
        lstm_cnn.history, "LSTM-CNN (Hybrid)",
        save_path=os.path.join(config["results"]["figures_path"], "lstm_cnn_history.png")
    )
    
    # ROC data
    y_proba = lstm_cnn.predict_proba(X_test)
    roc_data["LSTM-CNN (Hybrid)"] = {"y_true": y_test, "y_proba": y_proba}
    
    # Save model
    lstm_cnn.save(config["results"]["models_path"])
    
    # --- 3b. Baseline models ---
    baseline_names = config["model"]["baselines"]
    
    for model_name in baseline_names:
        print(f"\n{'='*60}")
        print(f"  STEP 3b: Training {model_name.upper()}")
        print(f"{'='*60}")
        
        bl_cfg = get_model_config(config, model_name)
        model = create_baseline(model_name, bl_cfg)
        model.build(input_shape)
        
        start = time.time()
        model.train(X_train, y_train, X_val, y_val)
        train_time = time.time() - start
        
        metrics = model.evaluate(X_test, y_test)
        metrics["training_time_sec"] = round(train_time, 2)
        all_results[model_name.upper()] = metrics
        print_results(model_name.upper(), metrics)
        
        # Training history
        if model.history:
            plot_training_history(
                model.history, model_name.upper(),
                save_path=os.path.join(config["results"]["figures_path"],
                                       f"{model_name}_history.png")
            )
        
        # ROC data
        y_proba = model.predict_proba(X_test)
        roc_data[model_name.upper()] = {"y_true": y_test, "y_proba": y_proba}
        
        # Save model
        model.save(config["results"]["models_path"])
    
    # ===== 4. GENERATE COMPARISON RESULTS =====
    print("\n" + "="*60)
    print("  STEP 4: Generating Results")
    print("="*60)
    
    # Comparison table
    comparison_df = compare_models(all_results)
    print("\n" + comparison_df.to_string())
    
    # Save table
    table_path = os.path.join(config["results"]["tables_path"], "model_comparison.csv")
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    comparison_df.to_csv(table_path)
    print(f"\n[Runner] Comparison table saved: {table_path}")
    
    # ROC curves
    plot_roc_curves(
        roc_data,
        save_path=os.path.join(config["results"]["figures_path"], "roc_curves.png")
    )
    
    # Comparison bar chart
    plot_comparison_bar(
        comparison_df,
        save_path=os.path.join(config["results"]["figures_path"], "comparison_bar.png")
    )
    
    # Confusion matrices for key models
    for name in ["LSTM-CNN (Hybrid)"]:
        y_pred = lstm_cnn.predict(X_test)
        plot_confusion_matrix(
            y_test, y_pred, title=f"{name} — Confusion Matrix",
            save_path=os.path.join(config["results"]["figures_path"],
                                   f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
        )
    
    # Save all results as JSON
    results_json = {}
    for k, v in all_results.items():
        results_json[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                           for kk, vv in v.items()}
    
    json_path = os.path.join(config["results"]["tables_path"], "all_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"[Runner] Full results saved: {json_path}")
    
    print("\n" + "="*60)
    print("  ALL EXPERIMENTS COMPLETE!")
    print("="*60)
    
    return all_results


def run_single(model_name, config_path=None, sample_frac=None):
    """Run a single model experiment (for debugging or quick testing).
    
    Args:
        model_name: Model to run ('lstm_cnn', 'cnn', 'rnn', 'lstm', 'bilstm', 'gru').
        config_path: Path to config.yaml.
        sample_frac: Sample fraction for quick testing.
    
    Returns:
        dict: Metrics for the model.
    """
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    
    download_dataset(dataset_cfg["kaggle_slug"], dataset_cfg["raw_path"])
    df = load_bot_iot(dataset_cfg["raw_path"], sample_frac=sample_frac)
    data = preprocess_pipeline(df, config)
    
    X_train = reshape_for_dl(data["X_train"])
    X_val = reshape_for_dl(data["X_val"])
    X_test = reshape_for_dl(data["X_test"])
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model_cfg = get_model_config(config, model_name)
    
    if model_name == "lstm_cnn":
        model = LSTMCNNDetector(model_cfg)
    else:
        model = create_baseline(model_name, model_cfg)
    
    model.build(input_shape)
    model.train(X_train, data["y_train"], X_val, data["y_val"])
    
    metrics = model.evaluate(X_test, data["y_test"])
    print_results(model_name, metrics)
    
    return metrics


# Allow running from command line
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MITM Detection Experiment Runner")
    parser.add_argument("--model", type=str, default=None,
                        help="Run single model (lstm_cnn, cnn, rnn, lstm, bilstm, gru)")
    parser.add_argument("--sample", type=float, default=None,
                        help="Sample fraction (e.g., 0.1 for 10%% of data)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")
    
    args = parser.parse_args()
    
    if args.model:
        run_single(args.model, config_path=args.config, sample_frac=args.sample)
    else:
        run_all(config_path=args.config, sample_frac=args.sample)

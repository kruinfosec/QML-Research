"""
Configuration loader for the MITM Research Implementation Platform.
Reads config.yaml and provides a centralized config dict to all modules.
"""
import os
import yaml


def load_config(path=None):
    """Load configuration from YAML file.
    
    Args:
        path: Path to config.yaml. If None, looks in project root.
    
    Returns:
        dict: Configuration dictionary.
    """
    if path is None:
        # Walk up from this file to find config.yaml in project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(project_root, "config.yaml")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    # Ensure output directories exist
    for dir_key in ["figures_path", "tables_path", "models_path", "logs_path"]:
        dir_path = config.get("results", {}).get(dir_key)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
    
    # Ensure data directories exist
    for dir_key in ["raw_path", "processed_path"]:
        dir_path = config.get("dataset", {}).get(dir_key)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
    
    return config


def get_model_config(config, model_name):
    """Get configuration for a specific model.
    
    Args:
        config: Full config dict.
        model_name: Name of the model (e.g., 'lstm_cnn', 'cnn', 'rnn').
    
    Returns:
        dict: Model-specific configuration.
    """
    if model_name == "lstm_cnn":
        return config["model"]["lstm_cnn"]
    elif model_name in config["model"]["baselines"]:
        return config["model"]["baselines"][model_name]
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: lstm_cnn, {list(config['model']['baselines'].keys())}")

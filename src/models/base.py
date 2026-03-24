"""
Abstract base class for all detection models.
Enforces a consistent interface across classical ML and deep learning models.
"""
import os
import numpy as np
from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """Abstract base for all intrusion detection models."""
    
    def __init__(self, name, config):
        """
        Args:
            name: Model name (e.g., 'lstm_cnn', 'xgboost').
            config: Model-specific config dict.
        """
        self.name = name
        self.config = config
        self.model = None
        self.history = None  # Training history (for DL models)
    
    @abstractmethod
    def build(self, input_shape, n_classes=1):
        """Build the model architecture.
        
        Args:
            input_shape: Shape of input features (excluding batch dim).
            n_classes: Number of output classes (1 for binary).
        """
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model.
        
        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data (optional).
        
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """Generate predictions.
        
        Args:
            X: Feature matrix.
        
        Returns:
            numpy array: Predicted labels.
        """
        pass
    
    def predict_proba(self, X):
        """Generate prediction probabilities (for ROC/AUC).
        
        Args:
            X: Feature matrix.
        
        Returns:
            numpy array: Predicted probabilities.
        """
        # Default: return predictions as probabilities
        return self.predict(X).astype(float)
    
    def evaluate(self, X, y):
        """Evaluate model on test data.
        
        Args:
            X: Feature matrix.
            y: True labels.
        
        Returns:
            dict: Evaluation metrics.
        """
        from src.experiments.metrics import compute_metrics
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        return compute_metrics(y, y_pred, y_proba)
    
    def save(self, path):
        """Save model to disk.
        
        Args:
            path: Directory to save to.
        """
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f"{self.name}_model")
        
        if hasattr(self.model, "save"):
            # Keras model
            self.model.save(save_path + ".keras")
        else:
            # Sklearn model
            import joblib
            joblib.dump(self.model, save_path + ".joblib")
        
        print(f"[{self.name}] Model saved to {save_path}")
    
    def summary(self):
        """Print model summary."""
        if hasattr(self.model, "summary"):
            self.model.summary()
        else:
            print(f"[{self.name}] Model: {type(self.model).__name__}")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"

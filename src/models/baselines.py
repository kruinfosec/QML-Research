"""
Baseline Deep Learning Models for comparison — Paper 1 Table 4.

Implements 5 standalone models to benchmark against the Hybrid LSTM-CNN:
1. CNN (2 Conv1D layers, 128 filters)
2. Simple RNN (2 layers, 128 units)
3. LSTM (2 layers, 128 units)
4. BiLSTM (2 Bidirectional LSTM layers, 128 units)
5. GRU (2 layers, 128 units)

All use Adam optimizer, ReLU activation, sigmoid output, binary cross-entropy.
"""
import numpy as np
from src.models.base import BaseDetector


class _DLBaseline(BaseDetector):
    """Generic deep learning baseline builder."""
    
    def build(self, input_shape, n_classes=1):
        """Build the model. Subclasses define _build_layers()."""
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        cfg = self.config
        inputs = layers.Input(shape=input_shape)
        
        x = self._build_layers(inputs, cfg)
        
        # Dense + Output
        x = layers.Flatten(name="flatten")(x) if len(x.shape) > 2 else x
        x = layers.Dense(64, activation="relu", name="dense_1")(x)
        x = layers.Dropout(cfg.get("dropout", 0.3), name="dropout_out")(x)
        
        if n_classes == 1:
            outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
            loss = "binary_crossentropy"
        else:
            outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)
            loss = "categorical_crossentropy"
        
        self.model = models.Model(inputs=inputs, outputs=outputs, name=self.name)
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.get("learning_rate", 0.001)
        )
        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        
        print(f"\n[{self.name}] Model built: {self.model.count_params():,} parameters")
        return self
    
    def _build_layers(self, inputs, cfg):
        raise NotImplementedError
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        import tensorflow as tf
        
        cfg = self.config
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5,
                restore_best_weights=True, verbose=1
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        print(f"[{self.name}] Training for up to {cfg.get('epochs', 50)} epochs...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=cfg.get("epochs", 50),
            batch_size=cfg.get("batch_size", 128),
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        return self
    
    def predict(self, X):
        y_proba = self.model.predict(X, verbose=0)
        if y_proba.shape[-1] == 1:
            return (y_proba.flatten() > 0.5).astype(int)
        return np.argmax(y_proba, axis=1)
    
    def predict_proba(self, X):
        proba = self.model.predict(X, verbose=0)
        return proba.flatten() if proba.shape[-1] == 1 else proba


class CNNDetector(_DLBaseline):
    """Basic CNN baseline (Paper 1 comparison)."""
    
    def __init__(self, config):
        super().__init__(name="cnn", config=config)
    
    def _build_layers(self, inputs, cfg):
        from tensorflow.keras import layers
        
        filters_list = cfg.get("filters", [64, 128])
        kernel_size = cfg.get("kernel_size", 3)
        dropout = cfg.get("dropout", 0.3)
        
        x = inputs
        for i, filters in enumerate(filters_list):
            x = layers.Conv1D(filters, kernel_size, activation="relu",
                              padding="same", name=f"conv_{i+1}")(x)
            x = layers.MaxPooling1D(2, name=f"pool_{i+1}")(x)
        
        x = layers.Dropout(dropout, name="cnn_dropout")(x)
        return x


class RNNDetector(_DLBaseline):
    """Simple RNN baseline (Paper 1 comparison)."""
    
    def __init__(self, config):
        super().__init__(name="rnn", config=config)
    
    def _build_layers(self, inputs, cfg):
        from tensorflow.keras import layers
        
        n_layers = cfg.get("layers", 2)
        units = cfg.get("units", 128)
        dropout = cfg.get("dropout", 0.3)
        
        x = inputs
        for i in range(n_layers):
            return_sequences = (i < n_layers - 1)
            x = layers.SimpleRNN(units, return_sequences=return_sequences,
                                 name=f"rnn_{i+1}")(x)
            x = layers.Dropout(dropout, name=f"rnn_dropout_{i+1}")(x)
        return x


class LSTMDetector(_DLBaseline):
    """Standalone LSTM baseline (Paper 1 comparison)."""
    
    def __init__(self, config):
        super().__init__(name="lstm", config=config)
    
    def _build_layers(self, inputs, cfg):
        from tensorflow.keras import layers
        
        n_layers = cfg.get("layers", 2)
        units = cfg.get("units", 128)
        dropout = cfg.get("dropout", 0.3)
        
        x = inputs
        for i in range(n_layers):
            return_sequences = (i < n_layers - 1)
            x = layers.LSTM(units, return_sequences=return_sequences,
                            name=f"lstm_{i+1}")(x)
            x = layers.Dropout(dropout, name=f"lstm_dropout_{i+1}")(x)
        return x


class BiLSTMDetector(_DLBaseline):
    """Bidirectional LSTM baseline (Paper 1 comparison)."""
    
    def __init__(self, config):
        super().__init__(name="bilstm", config=config)
    
    def _build_layers(self, inputs, cfg):
        from tensorflow.keras import layers
        
        n_layers = cfg.get("layers", 2)
        units = cfg.get("units", 128)
        dropout = cfg.get("dropout", 0.3)
        
        x = inputs
        for i in range(n_layers):
            return_sequences = (i < n_layers - 1)
            x = layers.Bidirectional(
                layers.LSTM(units, return_sequences=return_sequences),
                name=f"bilstm_{i+1}"
            )(x)
            x = layers.Dropout(dropout, name=f"bilstm_dropout_{i+1}")(x)
        return x


class GRUDetector(_DLBaseline):
    """GRU baseline (Paper 1 comparison)."""
    
    def __init__(self, config):
        super().__init__(name="gru", config=config)
    
    def _build_layers(self, inputs, cfg):
        from tensorflow.keras import layers
        
        n_layers = cfg.get("layers", 2)
        units = cfg.get("units", 128)
        dropout = cfg.get("dropout", 0.2)  # GRU uses 0.2 per Paper 1 Table 4
        
        x = inputs
        for i in range(n_layers):
            return_sequences = (i < n_layers - 1)
            x = layers.GRU(units, return_sequences=return_sequences,
                           name=f"gru_{i+1}")(x)
            x = layers.Dropout(dropout, name=f"gru_dropout_{i+1}")(x)
        return x


# ---- Factory function ----

def create_baseline(model_name, config):
    """Create a baseline model by name.
    
    Args:
        model_name: One of 'cnn', 'rnn', 'lstm', 'bilstm', 'gru'.
        config: Model-specific config dict.
    
    Returns:
        BaseDetector: Instantiated model (not yet built).
    """
    models = {
        "cnn": CNNDetector,
        "rnn": RNNDetector,
        "lstm": LSTMDetector,
        "bilstm": BiLSTMDetector,
        "gru": GRUDetector,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown baseline: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](config)

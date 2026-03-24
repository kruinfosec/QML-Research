"""
Hybrid LSTM-CNN Model — Paper 1 Replication.

Architecture from Sinha et al., 2025:
"A High Performance Hybrid LSTM-CNN Secure Architecture for IoT Environments"

Input → LSTM(256, tanh) × 3 layers → Conv1D(64,128,256) → MaxPool → Dense → Sigmoid

Hyperparameters from Tables 4 & 5:
- LSTM: 3 layers, 256 units, tanh activation, dropout 0.3
- CNN: 3 conv layers (64, 128, 256 filters), kernel 3, MaxPool 2
- Training: Adam(lr=0.0005), binary cross-entropy, 50 epochs, batch 128
- Regularization: Dropout 0.3, L2, EarlyStopping, gradient clipping
"""
import numpy as np
from src.models.base import BaseDetector


class LSTMCNNDetector(BaseDetector):
    """Hybrid LSTM-CNN model for intrusion detection (Paper 1)."""
    
    def __init__(self, config):
        super().__init__(name="lstm_cnn", config=config)
    
    def build(self, input_shape, n_classes=1):
        """Build the Hybrid LSTM-CNN architecture.
        
        Args:
            input_shape: (timesteps, features) — e.g. (1, 43)
            n_classes: 1 for binary classification.
        """
        import tensorflow as tf
        from tensorflow.keras import layers, models, regularizers
        
        cfg = self.config
        l2_reg = regularizers.l2(cfg.get("l2_reg", 0.001))
        
        # Input layer
        inputs = layers.Input(shape=input_shape)
        
        # ===== LSTM Block (Temporal Feature Extraction) =====
        x = inputs
        lstm_layers = cfg.get("lstm_layers", 3)
        lstm_units = cfg.get("lstm_units", 256)
        lstm_activation = cfg.get("lstm_activation", "tanh")
        dropout_rate = cfg.get("dropout", 0.3)
        
        for i in range(lstm_layers):
            return_sequences = True  # All LSTM layers return sequences for CNN input
            x = layers.LSTM(
                units=lstm_units,
                activation=lstm_activation,
                return_sequences=return_sequences,
                kernel_regularizer=l2_reg,
                name=f"lstm_{i+1}"
            )(x)
            x = layers.Dropout(dropout_rate, name=f"lstm_dropout_{i+1}")(x)
        
        # ===== CNN Block (Spatial Feature Extraction) =====
        cnn_filters = cfg.get("cnn_filters", [64, 128, 256])
        kernel_size = cfg.get("kernel_size", 3)
        
        for i, filters in enumerate(cnn_filters):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                kernel_regularizer=l2_reg,
                name=f"conv1d_{i+1}"
            )(x)
        
        # GlobalMaxPooling (handles any temporal dimension, including 1)
        x = layers.GlobalMaxPooling1D(name="global_maxpool")(x)
        
        # ===== Dense Block (Classification) =====
        
        dense_units = cfg.get("dense_units", 128)
        x = layers.Dense(dense_units, activation="relu", kernel_regularizer=l2_reg, name="dense_1")(x)
        x = layers.Dropout(dropout_rate, name="dense_dropout")(x)
        
        # Output
        output_activation = cfg.get("output_activation", "sigmoid")
        if n_classes == 1:
            outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
        else:
            outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs, name="LSTM_CNN_Hybrid")
        
        # Compile
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.get("learning_rate", 0.0005),
            clipnorm=1.0  # Gradient clipping (Paper 1)
        )
        
        loss = cfg.get("loss", "binary_crossentropy")
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy"]
        )
        
        print(f"\n[LSTM-CNN] Model built: {self.model.count_params():,} parameters")
        self.model.summary()
        
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model with EarlyStopping and ModelCheckpoint.
        
        Args:
            X_train: Training features (3D: samples, timesteps, features).
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
        
        Returns:
            self
        """
        import tensorflow as tf
        
        cfg = self.config
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.get("early_stopping_patience", 5),
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        print(f"\n[LSTM-CNN] Training for up to {cfg.get('epochs', 50)} epochs...")
        
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
        """Generate binary predictions."""
        y_proba = self.model.predict(X, verbose=0)
        if y_proba.shape[-1] == 1:
            return (y_proba.flatten() > 0.5).astype(int)
        else:
            return np.argmax(y_proba, axis=1)
    
    def predict_proba(self, X):
        """Generate prediction probabilities."""
        proba = self.model.predict(X, verbose=0)
        return proba.flatten() if proba.shape[-1] == 1 else proba

"""Коллбэки для обучения"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Optional

class ModelCheckpoint(keras.callbacks.Callback):
    """Сохранение лучшей модели"""
    
    def __init__(self, filepath: str, monitor: str = 'val_accuracy', mode: str = 'max'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best = -np.Inf if mode == 'max' else np.Inf
        
    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
            
        if (self.mode == 'max' and current > self.best) or \
           (self.mode == 'min' and current < self.best):
            self.best = current
            self.model.save(self.filepath)
            print(f"\n✅ Model saved to {self.filepath} (improved to {current:.4f})")


def get_default_callbacks(patience: int = 10, model_path: str = 'best_model.h5'):
    """
    Получить стандартные коллбэки для обучения
    
    Args:
        patience: терпеливость для early stopping
        model_path: путь для сохранения модели
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            mode='max'
        ),
        keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True
        )
    ]
    return callbacks

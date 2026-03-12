"""Обучение модели"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Dict, Any, Optional, List
from ..models.base_model import BaseModel
from ..evaluation.metrics import MetricsCalculator

class ModelTrainer:
    """Тренер для обучения модели"""
    
    def __init__(self, 
                 model: BaseModel,
                 config: Dict[str, Any],
                 callbacks: Optional[List[keras.callbacks.Callback]] = None):
        """
        Args:
            model: модель для обучения
            config: конфигурация обучения
            callbacks: список коллбэков
        """
        self.model = model
        self.config = config
        self.callbacks = callbacks or []
        self.history = None
        
    def compile_model(self):
        """Скомпилировать модель"""
        self.model.compile(
            optimizer=self.config.get('optimizer', 'adam'),
            loss=self.config.get('loss', 'categorical_crossentropy'),
            metrics=self.config.get('metrics', ['accuracy'])
        )
    
    def train(self, 
              X_sensor_train: np.ndarray,
              X_img_train: np.ndarray,
              y_train: np.ndarray,
              X_sensor_val: Optional[np.ndarray] = None,
              X_img_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, list]:
        """
        Обучить модель
        
        Args:
            X_sensor_train: тренировочные данные сенсоров
            X_img_train: тренировочные изображения
            y_train: тренировочные метки
            X_sensor_val: валидационные данные сенсоров
            X_img_val: валидационные изображения
            y_val: валидационные метки
            
        Returns:
            история обучения
        """
        # Компилируем если ещё нет
        if self.model.model is None:
            self.compile_model()
        
        # Подготовка данных для валидации
        validation_data = None
        if X_sensor_val is not None and X_img_val is not None and y_val is not None:
            validation_data = ([X_img_val, X_sensor_val], y_val)
        
        # Обучение
        self.history = self.model.model.fit(
            [X_img_train, X_sensor_train],
            y_train,
            batch_size=self.config.get('batch_size', 32),
            epochs=self.config.get('epochs', 50),
            validation_data=validation_data,
            validation_split=0.2 if validation_data is None else None,
            callbacks=self.callbacks,
            verbose=1,
            shuffle=True
        )
        
        return self.history.history
    
    def evaluate(self,
                 X_sensor_test: np.ndarray,
                 X_img_test: np.ndarray,
                 y_test: np.ndarray) -> Dict[str, float]:
        """
        Оценить модель на тестовых данных
        
        Args:
            X_sensor_test: тестовые данные сенсоров
            X_img_test: тестовые изображения
            y_test: тестовые метки
            
        Returns:
            словарь с метриками
        """
        if self.model.model is None:
            raise ValueError("Model not trained yet")
        
        # Оценка
        loss, accuracy = self.model.model.evaluate(
            [X_img_test, X_sensor_test],
            y_test,
            verbose=0
        )
        
        # Получаем предсказания для дополнительных метрик
        y_pred = self.model.model.predict([X_img_test, X_sensor_test])
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Вычисляем дополнительные метрики
        metrics = MetricsCalculator()
        additional_metrics = metrics.compute_all(y_true_classes, y_pred_classes)
        
        result = {
            'test_loss': loss,
            'test_accuracy': accuracy,
            **additional_metrics
        }
        
        return result
    
    def predict(self,
                X_sensor: np.ndarray,
                X_img: np.ndarray) -> np.ndarray:
        """
        Сделать предсказание
        
        Args:
            X_sensor: данные сенсоров
            X_img: изображения
            
        Returns:
            вероятности классов
        """
        if self.model.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.model.predict([X_img, X_sensor])

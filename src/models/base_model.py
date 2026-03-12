"""Базовый класс для всех моделей"""
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, Optional

class BaseModel(ABC):
    """Абстрактный базовый класс для моделей"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model: Optional[keras.Model] = None
        
    @abstractmethod 
    def build(self) -> keras.Model:
        """Построить архитектуру модели"""
        pass
    
    def compile(self, optimizer: str = 'adam', 
                loss: str = 'categorical_crossentropy',
                metrics: list = ['accuracy']):
        """Скомпилировать модель"""
        if self.model is None:
            self.model = self.build()
            
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def summary(self):
        """Вывести структуру модели"""
        if self.model is None:
            self.model = self.build()
        self.model.summary()
    
    def save(self, path: str):
        """Сохранить модель"""
        if self.model is None:
            raise ValueError("Model not built yet")
        self.model.save(path)
    
    def load(self, path: str):
        """Загрузить модель"""
        self.model = keras.models.load_model(path)

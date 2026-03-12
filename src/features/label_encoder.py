"""Кодирование меток классов"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import List, Dict

class ClassLabelEncoder:
    """Encoder для меток классов"""
    
    def __init__(self):
        self.encoder = LabelEncoder()
        self.classes_ = None
        
    def fit(self, y: pd.Series) -> 'ClassLabelEncoder':
        """Обучить encoder на метках"""
        self.encoder.fit(y)
        self.classes_ = self.encoder.classes_
        return self
    
    def transform(self, y: pd.Series) -> np.ndarray:
        """Преобразовать метки в числа"""
        return self.encoder.transform(y)
    
    def fit_transform(self, y: pd.Series) -> np.ndarray:
        """Обучить и преобразовать"""
        return self.encoder.fit_transform(y)
    
    def inverse_transform(self, y: np.ndarray) -> List[str]:
        """Преобразовать числа обратно в метки"""
        return self.encoder.inverse_transform(y)
    
    def get_class_mapping(self) -> Dict[int, str]:
        """Получить словарь {id: class_name}"""
        return dict(enumerate(self.classes_))
    
    def save(self, path: str):
        """Сохранить encoder"""
        joblib.dump(self.encoder, path)
        
    def load(self, path: str):
        """Загрузить encoder"""
        self.encoder = joblib.load(path)
        self.classes_ = self.encoder.classes_


def to_categorical(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Преобразовать метки в one-hot encoding
    
    Args:
        y: метки (N,)
        num_classes: количество классов
        
    Returns:
        one-hot матрица (N, num_classes)
    """
    from tensorflow.keras.utils import to_categorical as keras_to_categorical
    return keras_to_categorical(y, num_classes=num_classes)

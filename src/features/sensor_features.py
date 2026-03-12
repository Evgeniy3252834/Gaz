"""Обработка данных сенсоров"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional, List
import joblib

class SensorFeatureExtractor:
    """Извлечение и нормализация признаков сенсоров"""
    
    def __init__(self, sensor_columns: List[str]):
        """
        Args:
            sensor_columns: список названий колонок сенсоров
        """
        self.sensor_columns = sensor_columns
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'SensorFeatureExtractor':
        """
        Обучить нормализатор на данных
        
        Args:
            df: DataFrame с данными сенсоров
        """
        sensor_data = df[self.sensor_columns].values
        self.scaler.fit(sensor_data)
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Применить нормализацию
        
        Args:
            df: DataFrame с данными сенсоров
            
        Returns:
            нормализованные данные (N, 8)
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        sensor_data = df[self.sensor_columns].values
        return self.scaler.transform(sensor_data)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Обучить и применить за один шаг"""
        sensor_data = df[self.sensor_columns].values
        return self.scaler.fit_transform(sensor_data)
    
    def save(self, path: str):
        """Сохранить нормализатор"""
        joblib.dump(self.scaler, path)
        
    def load(self, path: str):
        """Загрузить нормализатор"""
        self.scaler = joblib.load(path)
        self.is_fitted = True


class SensorStatistics:
    """Статистика по сенсорам"""
    
    @staticmethod
    def print_sensor_stats(df: pd.DataFrame, sensor_columns: List[str]):
        """Вывести статистику по каждому сенсору"""
        print("\n📊 Sensor Statistics:")
        for col in sensor_columns:
            stats = df[col].describe()
            print(f"\n{col}:")
            print(f"  - Min: {stats['min']:.2f}")
            print(f"  - Max: {stats['max']:.2f}")
            print(f"  - Mean: {stats['mean']:.2f}")
            print(f"  - Std: {stats['std']:.2f}")

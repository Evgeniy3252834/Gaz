"""Сохранение и загрузка моделей и артефактов"""
import os
import joblib
import tensorflow as tf
from tensorflow import keras
from typing import Any, Dict, Optional
import json
import yaml

class ModelSaver:
    """Сохранение модели и связанных артефактов"""
    
    def __init__(self, base_path: str = 'models'):
        """
        Args:
            base_path: базовая директория для сохранения
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save_model(self, model: keras.Model, name: str = 'model'):
        """
        Сохранить Keras модель
        
        Args:
            model: модель для сохранения
            name: имя модели
        """
        path = os.path.join(self.base_path, f'{name}.h5')
        model.save(path)
        print(f"✅ Model saved to {path}")
        return path
    
    def save_scaler(self, scaler: Any, name: str = 'scaler'):
        """
        Сохранить StandardScaler
        
        Args:
            scaler: объект scaler
            name: имя файла
        """
        path = os.path.join(self.base_path, f'{name}.pkl')
        joblib.dump(scaler, path)
        print(f"✅ Scaler saved to {path}")
        return path
    
    def save_encoder(self, encoder: Any, name: str = 'label_encoder'):
        """
        Сохранить LabelEncoder
        
        Args:
            encoder: объект encoder
            name: имя файла
        """
        path = os.path.join(self.base_path, f'{name}.pkl')
        joblib.dump(encoder, path)
        print(f"✅ Encoder saved to {path}")
        return path
    
    def save_config(self, config: Dict, name: str = 'config'):
        """
        Сохранить конфигурацию в JSON
        
        Args:
            config: словарь с конфигурацией
            name: имя файла
        """
        path = os.path.join(self.base_path, f'{name}.json')
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✅ Config saved to {path}")
        return path
    
    def save_training_history(self, history: Dict, name: str = 'history'):
        """
        Сохранить историю обучения
        
        Args:
            history: история из model.fit()
            name: имя файла
        """
        path = os.path.join(self.base_path, f'{name}.json')
        
        # Конвертируем numpy значения в обычные числа
        serializable_history = {}
        for key, values in history.items():
            serializable_history[key] = [float(v) for v in values]
        
        with open(path, 'w') as f:
            json.dump(serializable_history, f, indent=4)
        print(f"✅ Training history saved to {path}")
        return path
    
    def save_all(self, model: keras.Model, scaler: Any, encoder: Any, 
                config: Dict, history: Dict, experiment_name: str):
        """
        Сохранить всё вместе в отдельную папку эксперимента
        
        Args:
            model: модель
            scaler: нормализатор
            encoder: кодировщик меток
            config: конфигурация
            history: история обучения
            experiment_name: название эксперимента
        """
        # Создаём папку для эксперимента
        exp_path = os.path.join(self.base_path, experiment_name)
        os.makedirs(exp_path, exist_ok=True)
        
        # Временно меняем base_path
        original_path = self.base_path
        self.base_path = exp_path
        
        # Сохраняем всё
        self.save_model(model, 'model')
        self.save_scaler(scaler, 'scaler')
        self.save_encoder(encoder, 'label_encoder')
        self.save_config(config, 'config')
        self.save_training_history(history, 'history')
        
        # Возвращаем обратно
        self.base_path = original_path
        
        print(f"\n✅ All artifacts saved to {exp_path}/")
        return exp_path


class ModelLoader:
    """Загрузка сохранённых моделей и артефактов"""
    
    def __init__(self, base_path: str = 'models'):
        self.base_path = base_path
    
    def load_model(self, name: str = 'model') -> keras.Model:
        """Загрузить Keras модель"""
        path = os.path.join(self.base_path, f'{name}.h5')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        return keras.models.load_model(path)
    
    def load_scaler(self, name: str = 'scaler'):
        """Загрузить StandardScaler"""
        path = os.path.join(self.base_path, f'{name}.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler not found: {path}")
        return joblib.load(path)
    
    def load_encoder(self, name: str = 'label_encoder'):
        """Загрузить LabelEncoder"""
        path = os.path.join(self.base_path, f'{name}.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Encoder not found: {path}")
        return joblib.load(path)
    
    def load_config(self, name: str = 'config') -> Dict:
        """Загрузить конфигурацию"""
        path = os.path.join(self.base_path, f'{name}.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, 'r') as f:
            return json.load(f)
    
    def load_training_history(self, name: str = 'history') -> Dict:
        """Загрузить историю обучения"""
        path = os.path.join(self.base_path, f'{name}.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"History not found: {path}")
        with open(path, 'r') as f:
            return json.load(f)
    
    def load_experiment(self, experiment_name: str) -> Dict:
        """
        Загрузить всё из эксперимента
        
        Returns:
            словарь с model, scaler, encoder, config, history
        """
        exp_path = os.path.join(self.base_path, experiment_name)
        
        # Временно меняем base_path
        original_path = self.base_path
        self.base_path = exp_path
        
        result = {
            'model': self.load_model('model'),
            'scaler': self.load_scaler('scaler'),
            'encoder': self.load_encoder('label_encoder'),
            'config': self.load_config('config'),
            'history': self.load_training_history('history')
        }
        
        # Возвращаем обратно
        self.base_path = original_path
        
        return result

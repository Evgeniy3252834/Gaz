"""MCNN (Multi-Channel Neural Network) модель для классификации газов"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Any, Tuple, List
from .base_model import BaseModel

class MCNNModel(BaseModel):
    """
    Multi-Channel Neural Network с двумя входами:
    - Термальные изображения (128x128 grayscale) -> CNN ветка
    - Данные сенсоров (8 признаков) -> Dense ветка
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: словарь с параметрами модели
                - image_size: Tuple[int, int]
                - cnn_filters: List[int]
                - cnn_kernel_size: List[int]
                - dense_units: List[int]
                - dropout_rates: List[float]
                - num_classes: int
                - activation: str
        """
        super().__init__(config)
        
    def _build_cnn_branch(self, input_shape: Tuple[int, int, int]) -> tf.Tensor:
        """
        Построить CNN ветку для обработки изображений
        
        Args:
            input_shape: форма входного изображения (128, 128, 1)
            
        Returns:
            выходной тензор после CNN
        """
        img_input = layers.Input(shape=input_shape, name='image_input')
        
        x = img_input
        filters = self.config.get('cnn_filters', [32, 64])
        kernel_sizes = self.config.get('cnn_kernel_size', [3, 3])
        
        for i, (filters_count, kernel) in enumerate(zip(filters, kernel_sizes)):
            x = layers.Conv2D(
                filters_count, 
                kernel, 
                activation=self.config.get('activation', 'relu'),
                name=f'conv_{i+1}'
            )(x)
            x = layers.MaxPooling2D(2, 2, name=f'pool_{i+1}')(x)
        
        x = layers.Flatten(name='cnn_flatten')(x)
        return x, img_input
    
    def _build_sensor_branch(self, input_shape: int) -> tf.Tensor:
        """
        Построить Dense ветку для данных сенсоров
        
        Args:
            input_shape: количество сенсоров (8)
            
        Returns:
            выходной тензор после dense слоёв
        """
        sensor_input = layers.Input(shape=(input_shape,), name='sensor_input')
        
        x = sensor_input
        dense_units = self.config.get('dense_units', [64])
        dropout_rates = self.config.get('dropout_rates', [0.3])
        
        for i, (units, dropout) in enumerate(zip(dense_units, dropout_rates)):
            x = layers.Dense(
                units, 
                activation=self.config.get('activation', 'relu'),
                name=f'dense_{i+1}'
            )(x)
            if dropout > 0:
                x = layers.Dropout(dropout, name=f'dropout_{i+1}')(x)
        
        return x, sensor_input
    
    def build(self) -> keras.Model:
        """
        Построить полную MCNN модель
        
        Returns:
            скомпилированная Keras модель
        """
        # Параметры
        image_size = self.config.get('image_size', (128, 128))
        num_sensors = self.config.get('num_sensors', 8)
        num_classes = self.config.get('num_classes', 4)
        
        # Строим ветки
        cnn_output, img_input = self._build_cnn_branch((*image_size, 1))
        sensor_output, sensor_input = self._build_sensor_branch(num_sensors)
        
        # Конкатенация
        fused = layers.Concatenate(name='concat')([cnn_output, sensor_output])
        
        # Финальные слои
        final_dense_units = self.config.get('final_dense_units', [128])
        final_dropout = self.config.get('final_dropout', 0.5)
        
        x = fused
        for units in final_dense_units:
            x = layers.Dense(
                units, 
                activation=self.config.get('activation', 'relu'),
                name='final_dense'
            )(x)
        
        if final_dropout > 0:
            x = layers.Dropout(final_dropout, name='final_dropout')(x)
        
        output = layers.Dense(
            num_classes, 
            activation=self.config.get('output_activation', 'softmax'),
            name='output'
        )(x)
        
        # Создаём модель
        model = keras.Model(
            inputs=[img_input, sensor_input],
            outputs=output,
            name='MCNN_Gas_Classifier'
        )
        
        self.model = model
        return model
    
    def get_config(self) -> Dict[str, Any]:
        """Вернуть конфигурацию модели"""
        return self.config


def create_default_mcnn() -> MCNNModel:
    """
    Создать MCNN модель с параметрами по умолчанию
    
    Returns:
        экземпляр MCNNModel
    """
    default_config = {
        'image_size': (128, 128),
        'cnn_filters': [32, 64],
        'cnn_kernel_size': [3, 3],
        'dense_units': [64],
        'dropout_rates': [0.3],
        'final_dense_units': [128],
        'final_dropout': 0.5,
        'num_sensors': 8,
        'num_classes': 4,
        'activation': 'relu',
        'output_activation': 'softmax'
    }
    return MCNNModel(default_config)

# Version: 1.0.0
# Last updated: 2026-03-12

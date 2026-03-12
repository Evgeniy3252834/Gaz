#!/usr/bin/env python
"""Полный пайплайн обучения модели Gaz"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
from tensorflow.keras.utils import to_categorical

from src.data.loader import DataLoader, DataValidator
from src.data.preprocessor import DataPreprocessor
from src.features.sensor_features import SensorFeatureExtractor, SensorStatistics
from src.features.image_features import ImagePreprocessor
from src.features.label_encoder import ClassLabelEncoder
from src.models.mcnn_model import create_default_mcnn
from src.training.trainer import ModelTrainer
from src.training.callbacks import get_default_callbacks
from src.evaluation.metrics import MetricsCalculator, ConfusionMatrixDisplay
from src.evaluation.visualizer import TrainingVisualizer, PredictionVisualizer
from src.persistence.model_saver import ModelSaver

def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Загрузить конфигурацию"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Основной пайплайн"""
    print("="*60)
    print("🚀 Gaz Training Pipeline Started")
    print("="*60)
    
    # 1. Загружаем конфиг
    print("\n📥 Loading configuration...")
    config = load_config()
    
    # 2. Загружаем данные
    print("\n📊 Loading data...")
    loader = DataLoader(
        csv_path=config['data']['csv_path'],
        images_dir=config['data']['images_dir'],
        image_size=tuple(config['data']['image_size'])
    )
    
    df, images = loader.load_dataset()
    DataValidator.print_stats(df, images)
    
    # 3. Извлекаем признаки
    print("\n🔧 Extracting features...")
    
    # Сенсоры
    sensor_extractor = SensorFeatureExtractor(config['data']['sensor_columns'])
    sensor_data = sensor_extractor.fit_transform(df)
    SensorStatistics.print_sensor_stats(df, config['data']['sensor_columns'])
    
    # Метки
    label_encoder = ClassLabelEncoder()
    labels = label_encoder.fit_transform(df['class'])
    class_names = label_encoder.encoder.classes_
    print(f"\n📋 Classes: {class_names}")
    print(f"Class mapping: {label_encoder.get_class_mapping()}")
    
    # 4. Разделяем данные
    print("\n✂️ Splitting data...")
    preprocessor = DataPreprocessor(
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    split_data = preprocessor.split_data(sensor_data, images, labels)
    
    # One-hot encoding для меток
    num_classes = config['data']['num_classes']
    y_train_cat = to_categorical(split_data['y_train'], num_classes)
    y_test_cat = to_categorical(split_data['y_test'], num_classes)
    
    print(f"Train samples: {len(split_data['y_train'])}")
    print(f"Test samples: {len(split_data['y_test'])}")
    
    # 5. Создаём модель
    print("\n🏗️ Building model...")
    model = create_default_mcnn()
    model.summary()
    
    # 6. Настраиваем обучение
    print("\n⚙️ Setting up training...")
    callbacks = get_default_callbacks(
        patience=config['training']['patience'],
        model_path=config['persistence']['model_path']
    )
    
    trainer = ModelTrainer(
        model=model,
        config=config['training'],
        callbacks=callbacks
    )
    
    # 7. Обучаем
    print("\n🎯 Training...")
    history = trainer.train(
        X_sensor_train=split_data['X_sensor_train'],
        X_img_train=split_data['X_img_train'],
        y_train=y_train_cat,
        X_sensor_val=split_data['X_sensor_test'],
        X_img_val=split_data['X_img_test'],
        y_val=y_test_cat
    )
    
    # 8. Оцениваем
    print("\n📊 Evaluating...")
    metrics = trainer.evaluate(
        X_sensor_test=split_data['X_sensor_test'],
        X_img_test=split_data['X_img_test'],
        y_test=y_test_cat
    )
    
    print("\n✅ Test Results:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    # 9. Визуализируем
    print("\n📈 Visualizing...")
    visualizer = TrainingVisualizer()
    visualizer.plot_history(history)
    
    # Матрица ошибок
    y_pred = trainer.predict(
        split_data['X_sensor_test'],
        split_data['X_img_test']
    )
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = split_data['y_test']
    
    cm_display = ConfusionMatrixDisplay()
    cm = MetricsCalculator.confusion_matrix(y_true_classes, y_pred_classes)
    cm_display.plot(cm, class_names)
    
    # 10. Сохраняем всё
    print("\n💾 Saving artifacts...")
    saver = ModelSaver()
    experiment_path = saver.save_all(
        model=model.model,
        scaler=sensor_extractor.scaler,
        encoder=label_encoder.encoder,
        config=config,
        history=history,
        experiment_name='experiment_1'
    )
    
    print("\n" + "="*60)
    print("✅ Training completed successfully!")
    print(f"📁 All artifacts saved to: {experiment_path}")
    print("="*60)

if __name__ == "__main__":
    main()

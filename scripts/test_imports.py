#!/usr/bin/env python
"""Проверка, что все модули импортируются правильно"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Проверить импорты всех модулей"""
    modules = [
        'src.data.loader',
        'src.data.preprocessor',
        'src.features.sensor_features',
        'src.features.image_features',
        'src.features.label_encoder',
        'src.models.mcnn_model',
        'src.training.trainer',
        'src.training.callbacks',
        'src.evaluation.metrics',
        'src.evaluation.visualizer',
        'src.persistence.model_saver'
    ]
    
    print("🔍 Testing imports...")
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
        except Exception as e:
            print(f"❌ {module_name}: {e}")
    
    print("\n🎯 All imports tested!")

if __name__ == "__main__":
    test_imports()

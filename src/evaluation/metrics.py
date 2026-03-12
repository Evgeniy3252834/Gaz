"""Метрики для оценки модели"""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
from typing import Dict, Tuple, List

class MetricsCalculator:
    """Вычисление метрик классификации"""
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Точность классификации"""
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
        """Precision (точность)"""
        return precision_score(y_true, y_pred, average=average)
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
        """Recall (полнота)"""
        return recall_score(y_true, y_pred, average=average)
    
    @staticmethod
    def f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
        """F1-score"""
        return f1_score(y_true, y_pred, average=average)
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Матрица ошибок"""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                            target_names: List[str] = None) -> str:
        """Полный отчёт по классификации"""
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def compute_all(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Вычислить все основные метрики"""
        return {
            'accuracy': self.accuracy(y_true, y_pred),
            'precision': self.precision(y_true, y_pred),
            'recall': self.recall(y_true, y_pred),
            'f1_score': self.f1(y_true, y_pred)
        }
    
    @staticmethod
    def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Метрики по каждому классу
        
        Args:
            y_true: истинные метки
            y_pred: предсказанные метки
            class_names: названия классов
            
        Returns:
            словарь с метриками для каждого класса
        """
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        result = {}
        for i, class_name in enumerate(class_names):
            result[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': int(support[i])
            }
        
        return result


class ConfusionMatrixDisplay:
    """Отображение матрицы ошибок"""
    
    @staticmethod
    def plot(cm: np.ndarray, class_names: List[str], save_path: str = None):
        """
        Построить тепловую карту матрицы ошибок
        
        Args:
            cm: матрица ошибок
            class_names: названия классов
            save_path: путь для сохранения (опционально)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

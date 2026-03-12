"""Визуализация результатов обучения"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional

class TrainingVisualizer:
    """Визуализация истории обучения"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        plt.style.use(style)
        
    def plot_history(self, history: Dict[str, List[float]], 
                    metrics: List[str] = ['accuracy', 'loss'],
                    save_path: Optional[str] = None):
        """
        Построить графики обучения
        
        Args:
            history: история обучения из model.fit()
            metrics: метрики для отображения
            save_path: путь для сохранения
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Обучающие данные
            train_values = history.get(metric, [])
            val_values = history.get(f'val_{metric}', [])
            epochs = range(1, len(train_values) + 1)
            
            ax.plot(epochs, train_values, 'b-', label=f'Training {metric}', linewidth=2)
            if val_values:
                ax.plot(epochs, val_values, 'r-', label=f'Validation {metric}', linewidth=2)
            
            ax.set_title(f'Model {metric.capitalize()}', fontsize=14)
            ax.set_xlabel('Epochs', fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_distribution(self, y_train: np.ndarray, y_test: np.ndarray,
                               class_names: List[str], save_path: Optional[str] = None):
        """
        Построить распределение классов
        
        Args:
            y_train: тренировочные метки
            y_test: тестовые метки
            class_names: названия классов
            save_path: путь для сохранения
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Тренировочные данные
        train_counts = np.bincount(y_train)
        ax1.bar(class_names, train_counts, color='skyblue', edgecolor='navy')
        ax1.set_title('Training Set Class Distribution', fontsize=14)
        ax1.set_xlabel('Classes', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Добавляем значения на столбцы
        for i, v in enumerate(train_counts):
            ax1.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        # Тестовые данные
        test_counts = np.bincount(y_test)
        ax2.bar(class_names, test_counts, color='lightcoral', edgecolor='darkred')
        ax2.set_title('Test Set Class Distribution', fontsize=14)
        ax2.set_xlabel('Classes', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(test_counts):
            ax2.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class PredictionVisualizer:
    """Визуализация предсказаний"""
    
    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                        class_names: List[str], num_samples: int = 10,
                        save_path: Optional[str] = None):
        """
        Показать примеры предсказаний
        
        Args:
            y_true: истинные метки
            y_pred: предсказанные метки
            class_names: названия классов
            num_samples: количество примеров
            save_path: путь для сохранения
        """
        indices = np.random.choice(len(y_true), num_samples, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices[:10]):  # максимум 10 примеров
            if i >= len(axes):
                break
                
            ax = axes[i]
            true_class = class_names[y_true[idx]]
            pred_class = class_names[y_pred[idx]]
            
            color = 'green' if y_true[idx] == y_pred[idx] else 'red'
            
            ax.text(0.5, 0.5, f'True: {true_class}\nPred: {pred_class}',
                   ha='center', va='center', fontsize=10,
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Sample {idx}')
        
        # Скрыть лишние подграфики
        for i in range(len(indices[:10]), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Prediction Examples', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

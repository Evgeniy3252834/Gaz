"""Сохранение результатов обучения"""
import os
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import numpy as np

class ResultsManager:
    """Менеджер для сохранения всех результатов обучения"""
    
    def __init__(self, base_dir: str = 'results'):
        """
        Args:
            base_dir: базовая папка для результатов
        """
        self.base_dir = base_dir
        self.plots_dir = os.path.join(base_dir, 'plots')
        self.metrics_dir = os.path.join(base_dir, 'metrics')
        self.models_dir = os.path.join(base_dir, 'models')
        
        # Создаём папки, если их нет
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def save_training_plots(self, history: Dict[str, List[float]], 
                           experiment_name: str = None):
        """
        Сохранить графики обучения
        
        Args:
            history: история из model.fit()
            experiment_name: название эксперимента (если None - создаст по времени)
        """
        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # График accuracy
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.get('accuracy', []), label='Train', linewidth=2)
        plt.plot(history.get('val_accuracy', []), label='Validation', linewidth=2)
        plt.title('Model Accuracy', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # График loss
        plt.subplot(1, 2, 2)
        plt.plot(history.get('loss', []), label='Train', linewidth=2)
        plt.plot(history.get('val_loss', []), label='Validation', linewidth=2)
        plt.title('Model Loss', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, f'{experiment_name}_training.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Training plots saved to {plot_path}")
        
        return plot_path
    
    def save_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                             experiment_name: str = None):
        """
        Сохранить матрицу ошибок
        
        Args:
            cm: матрица ошибок
            class_names: названия классов
            experiment_name: название эксперимента
        """
        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        plot_path = os.path.join(self.plots_dir, f'{experiment_name}_confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Confusion matrix saved to {plot_path}")
        
        return plot_path
    
    def save_metrics(self, metrics: Dict[str, float], 
                    experiment_name: str = None):
        """
        Сохранить метрики в JSON и CSV
        
        Args:
            metrics: словарь с метриками
            experiment_name: название эксперимента
        """
        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем в JSON
        json_path = os.path.join(self.metrics_dir, f'{experiment_name}_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"📈 Metrics saved to {json_path}")
        
        # Сохраняем в CSV
        csv_path = os.path.join(self.metrics_dir, f'{experiment_name}_metrics.csv')
        df = pd.DataFrame([metrics])
        df.to_csv(csv_path, index=False)
        print(f"📈 Metrics saved to {csv_path}")
        
        return json_path, csv_path
    
    def save_classification_report(self, report: str, class_names: List[str],
                                  experiment_name: str = None):
        """
        Сохранить classification report
        
        Args:
            report: строковый отчёт
            class_names: названия классов
            experiment_name: название эксперимента
        """
        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_path = os.path.join(self.metrics_dir, f'{experiment_name}_classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("="*50 + "\n")
            f.write(report)
        
        print(f"📄 Classification report saved to {report_path}")
        return report_path
    
    def save_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        class_names: List[str], experiment_name: str = None):
        """
        Сохранить предсказания в CSV
        
        Args:
            y_true: истинные метки
            y_pred: предсказанные метки
            class_names: названия классов
            experiment_name: название эксперимента
        """
        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Создаём DataFrame с результатами
        df = pd.DataFrame({
            'true_class_id': y_true,
            'predicted_class_id': y_pred,
            'true_class_name': [class_names[i] for i in y_true],
            'predicted_class_name': [class_names[i] for i in y_pred],
            'correct': y_true == y_pred
        })
        
        # Сохраняем
        csv_path = os.path.join(self.metrics_dir, f'{experiment_name}_predictions.csv')
        df.to_csv(csv_path, index=False)
        print(f"📄 Predictions saved to {csv_path}")
        
        # Сохраняем статистику ошибок
        errors = df[df['correct'] == False]
        if len(errors) > 0:
            error_path = os.path.join(self.metrics_dir, f'{experiment_name}_errors.csv')
            errors.to_csv(error_path, index=False)
            print(f"⚠️ Error cases saved to {error_path}")
        
        return csv_path
    
    def save_experiment_summary(self, 
                               config: Dict[str, Any],
                               metrics: Dict[str, float],
                               history: Dict[str, List[float]],
                               experiment_name: str = None):
        """
        Сохранить полный отчёт об эксперименте
        
        Args:
            config: конфигурация эксперимента
            metrics: итоговые метрики
            history: история обучения
            experiment_name: название эксперимента
        """
        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary = {
            'experiment_name': experiment_name,
            'timestamp': datetime.datetime.now().isoformat(),
            'config': config,
            'metrics': metrics,
            'history_summary': {
                'best_accuracy': max(history.get('accuracy', [0])),
                'best_val_accuracy': max(history.get('val_accuracy', [0])),
                'best_loss': min(history.get('loss', [0])),
                'best_val_loss': min(history.get('val_loss', [0])),
                'epochs': len(history.get('accuracy', []))
            }
        }
        
        summary_path = os.path.join(self.metrics_dir, f'{experiment_name}_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"📋 Experiment summary saved to {summary_path}")
        return summary_path

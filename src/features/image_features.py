"""Обработка термальных изображений"""
import numpy as np
import cv2
from typing import Tuple, Optional

class ImagePreprocessor:
    """Предобработка изображений"""
    
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        self.target_size = target_size
        
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Изменить размер изображения"""
        return cv2.resize(image, self.target_size)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Нормализовать в [0, 1]"""
        return image.astype(np.float32) / 255.0
    
    def add_channel(self, image: np.ndarray) -> np.ndarray:
        """Добавить канал (H, W) -> (H, W, 1)"""
        return image.reshape(*self.target_size, 1)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Полный пайплайн предобработки"""
        image = self.resize_image(image)
        image = self.normalize(image)
        image = self.add_channel(image)
        return image


class ImageAugmenter:
    """Аугментация изображений для увеличения выборки"""
    
    def __init__(self, rotation_range: int = 10, flip_prob: float = 0.5):
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
        
    def random_rotate(self, image: np.ndarray) -> np.ndarray:
        """Случайный поворот"""
        angle = np.random.randint(-self.rotation_range, self.rotation_range)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(image, M, (w, h))
    
    def random_flip(self, image: np.ndarray) -> np.ndarray:
        """Случайное отражение"""
        if np.random.random() < self.flip_prob:
            return cv2.flip(image, 1)  # горизонтальное отражение
        return image
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """Применить аугментацию"""
        image = self.random_rotate(image)
        image = self.random_flip(image)
        return image

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

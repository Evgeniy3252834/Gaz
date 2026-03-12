# Gaz - Классификация газов по данным сенсоров и термальным изображениям

## 🎯 Описание проекта
Проект для классификации типов газов на основе:
- 8 газовых сенсоров (MQ2, MQ3, MQ5, MQ6, MQ7, MQ8, MQ9, MQ135)
- Термальных изображений (128x128, grayscale)

## 🏗️ Архитектура проекта (Clean ML Architecture)
src/
├── data/ # Загрузка и предобработка данных
├── features/ # Извлечение признаков, нормализация
├── models/ # Определение архитектур нейросетей
├── training/ # Циклы обучения, коллбэки
├── evaluation/ # Метрики, визуализация
├── persistence/ # Сохранение/загрузка моделей
└── api/ # Инференс (для будущего API)

text

## 📊 Модель 
Используется **MCNN (Multi-Channel Neural Network)**:
- CNN ветка для обработки термальных изображений
- Dense ветка для данных сенсоров
- Конкатенация и финальная классификация на 4 класса

## 🚀 Быстрый старт

### Установка
git clone https://github.com/Евгений3252834/Gaz.git
cd Gaz
python -m venv venv
source venv/bin/activate  # или venv\Scripts\activate на Windows
pip install -r requirements.txt

Обучение модели
## 📥 Данные

Датасет **НЕ хранится в репозитории** из-за большого размера.

### Скачать данные
1. Скачай архив с данными по ссылке:  
   👉 **[Google Drive: Gaz Dataset](https://drive.google.com/drive/folders/15STjy6tIWdqzjneHGkpSsEGNRuI3amB9?usp=drive_link)**
2. Распакуй содержимое в папку `data/` проекта так, чтобы получилась структура:

python scripts/train_pipeline.py
📈 Результаты
Точность на тесте: ~95% (будет уточнено)

📚 Структура репозитория
Gaz/
├── data/               # Сырые данные (в .gitignore)
├── notebooks/          # Jupyter ноутбуки для экспериментов
├── src/                # Основной код
│   ├── data/           # Загрузка данных
│   ├── features/       # Признаки
│   ├── models/         # Модели
│   ├── training/       # Обучение
│   ├── evaluation/     # Оценка
│   ├── persistence/    # Сохранение
│   └── api/            # Инференс
├── tests/              # Тесты
├── config/             # Конфиги
├── scripts/            # Скрипты для запуска
└── requirements.txt    # Зависимости
📄 Лицензия
MIT

# Документация проекта Headliners #

## 1. Архитектура и методы ##

Модель: YOLO различных поколений
Тип задачи: Object Detection (обнаружение объектов)
Фреймворк: Ultralytics YOLO с CLI интерфейсом

Архитектура CNN для решения задачи object-detection

Backbone: Сверточная сеть для извлечения признаков
Neck: FPN (Feature Pyramid Network) для многомасштабной обработки
Head: Предсказание bounding boxes и классов

## 2. Данные ##

**Структура данных**



**Разделение данных**

python
def train_test_split(path, neg_path=None, split=0.2):
    # Разделение 80%/20% train/validation
    # Стратифицированное перемешивание с фиксированным random_seed=42

## 3. Обоснование выбора методов ##

Скорость:  
Точность:  

Препроцессинг

## 4. Процесс обучения ##

**Параметры обучения**

epochs: 
imgsz: 
batch: 

**Параметры инференса**

bash
task=detect 
mode=predict 
conf=0.25

**Обработка данных**
Аугментация данных

## 5. Методы оценки и результаты тестирования ##

**Основные метрики**
Precision: Точность обнаружения
Recall: Полнота обнаружения
F1-score: Гармоническое среднее
mAP: Mean Average Precision
ROC-AUC: Площадь под ROC-кривой

**Функции расчета метрик**

def calculate_iou(box1, box2):
    # Расчет Intersection over Union

def calculate_precision_recall(pred_boxes, true_boxes, iou_threshold=0.5):
    # Расчет precision и recall

def calculate_map(predictions, targets, iou_threshold=0.5):
    # Расчет mean Average Precision

def calculate_detection_roc_auc(predictions, targets, iou_threshold=0.5):
    # Расчет ROC-AUC для детекции

**Результаты тестирования**

text
Precision: X.XXX
Recall: X.XXX
F1: X.XXX
mAP: X.XXX
ROC-AUC: X.XXX

## 6. Заключение ##

Разработанная система демонстрирует высокую эффективность в обнаружении инородных тел на рентгенограммах ОГК. Использование архитектуры YOLO позволило достичь баланса между точностью и скоростью работы, что важно для клинического применения.


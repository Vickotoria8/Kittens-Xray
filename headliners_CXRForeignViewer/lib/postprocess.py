'''
    Методы, использованные для постобработки
    результата модели
    
    Methods used to postprocess data
    of model results
'''

import os
import cv2
import re
import pandas as pd

from pathlib import Path

from .. import config

def merge_txt_files(
        source_dir:str # Директория с выводом модели
):
    # Создаем директорию, если она не существует
    dir = Path(config.RESULT_PATH)
    dir.mkdir(exist_ok=True)
    lib_dir = Path(''.join([config.RESULT_PATH, 'labels/']))
    lib_dir.mkdir(exist_ok=True)
    
    # Получаем список txt-файлов в текущей директории
    files = list(Path(source_dir).glob("*.txt"))
    
    # Словарь для группировки файлов по базовому имени
    file_groups = {}
    
    # Шаблон для поиска файлов с суффиксом (2)
    pattern = re.compile(r"^(.*?)\s*\(\d+\)\s*$")
    
    for file in files:
        stem = file.stem  # Имя файла без расширения
        
        # Проверяем, является ли файл версией с номером
        match = pattern.match(stem)
        if match:
            base_name = match.group(1)
        else:
            base_name = stem
        
        # Добавляем файл в соответствующую группу
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append(file)
    
    # Обрабатываем группы файлов
    for base_name, group in file_groups.items():
        if len(group) > 1:
            # Сортируем: оригинал первый, затем версии с номерами
            group.sort(key=lambda x: x.stem)
            
            # Читаем содержимое всех файлов группы
            content = []
            for file in group:
                with open(file, 'r', encoding='utf-8') as f:
                    content.append(f.read())
            
            # Создаем имя результирующего файла
            output_file = lib_dir / f"{base_name}.txt"
            
            # Записываем объединенное содержимое
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content))
            
            # print(f"Объединен файл: {output_file}")


def draw_yolo_boxes(image_path, label_path, output_dir):
    # Создаем директорию для результатов если её нет
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        # print(f"Ошибка загрузки изображения: {image_path}")
        return
        
    img_height, img_width = image.shape[:2]
    
    # Читаем файл разметки
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        # print(f"Файл разметки не найден: {label_path}")
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, image)
        return

    # Обрабатываем каждую метку
    for line in lines:
        data = line.strip().split()
        if len(data) != 5:
            continue
            
        class_id, x_center, y_center, width, height = map(float, data)
        
        # Конвертируем нормализованные координаты в абсолютные
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height
        
        # Рассчитываем координаты углов прямоугольника
        x1 = int(x_center_abs - width_abs / 2)
        y1 = int(y_center_abs - height_abs / 2)
        x2 = int(x_center_abs + width_abs / 2)
        y2 = int(y_center_abs + height_abs / 2)
        
        # Рисуем прямоугольник
        color = (255, 0, 0)  # Синий цвет
        thickness = 5
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Добавляем подпись класса (опционально)
        label = "Foreign item"
        text_thickness = 2
        cv2.putText(image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, text_thickness)

    # Сохраняем результат
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    # print(f"Обработано: {output_path}")

def draw_boxes():
    # Укажите пути к вашим данным
    images_dir = config.TEMP_TEST_DATA_PATH  # Папка с изображениями
    labels_dir = config.OUTPUT_LABELS_PATH  # Папка с разметкой YOLO
    output_dir = config.RESULT_PATH  # Папка для результатов
    
    # Обрабатываем каждый файл
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg") and "(2)" not in filename:
            # Формируем пути к файлам
            image_path = os.path.join(images_dir, filename)
            label_name = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_name)
            
            # Обрабатываем изображение
            draw_yolo_boxes(image_path, label_path, output_dir)


def create_file_mapping_table(images_dir: str, labels_dir: str, output_excel: str = "file_mapping.xlsx") -> None:
    """
    Создает Excel таблицу с mapping файлов изображений и соответствующих txt файлов
    
    Args:
        images_dir: Папка с изображениями
        labels_dir: Папка с txt файлами (лейблами)
        output_excel: Путь для сохранения Excel файла
    """
    
    # Получаем списки файлов
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.dcm'))]
    label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]
    
    # Создаем множества имен без расширений для быстрого поиска
    image_names = {os.path.splitext(f)[0] for f in image_files}
    label_names = {os.path.splitext(f)[0] for f in label_files}
    
    # Создаем данные для таблицы
    data = []
    for image_name in sorted(image_names):
        has_label = 1 if image_name in label_names else 0
        data.append({
            'file_id': image_name,
            'has_label': has_label
        })
    
    # Создаем DataFrame
    df = pd.DataFrame(data)
    
    # Сохраняем в Excel
    df.to_excel(output_excel, index=False)
    

# Пример использования
def make_binary():
    images_dir = config.TEST_FILES_PATH
    labels_dir = ''.join([config.RESULT_PATH, 'labels/'])
    
    create_file_mapping_table(images_dir, labels_dir, 'result.xlsx')
'''
    Функции файлового менеджмента,
    использованные командой разработчиков
    для подготовки датасетов

    Methods of filed management used by
    development team to prepare train and test data
'''

import os
import shutil
from PIL import Image
import pydicom as dicom
import numpy as np
import re
from pathlib import Path

from image_processing import prepare_dicom_image, get_dicom_window_attributes

def make_empties(
        folder_path: str # Папка с файлами в формате .jpg / Folder with .jpg files
        ) -> None:
    
    '''
        Создать пустные файлы разметки (.txt)
        для изображений без файлов разметки

        Create empty label files (.txt)
        for images without label files
    '''
        
    # Обойти все файлы в папке
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            base_name = os.path.splitext(filename)[0]
            txt_file_path = os.path.join(folder_path, base_name + '.txt')
            
            # Проверить наличие .txt
            if not os.path.exists(txt_file_path):
                # Создать пустой .txt файл
                with open(txt_file_path, 'w') as f:
                    pass  # просто открыть и закрыть для создания пустого файла
                print(f'Создан пустой файл: {txt_file_path}')


def copy_txt(
        source_dir: str, # Директория с файлами / Directory with files
        target_dir: str # Директория для копирования / Directory for copying
        ) -> None:
    
    '''
        Скопировать файлы разметки в формате .txt
        из одной папки в другую

        Copy .txt files with labels
        from one folder to another
    '''

    # Убедитесь, что целевая папка существует
    os.makedirs(target_dir, exist_ok=True)

    # Обход всех файлов внутри source_dir
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.txt'):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)
                # Копируем файл
                shutil.copy2(source_path, target_path)
                print(f'Скопировал: {source_path} -> {target_path}')


def rename_files(
        folder_path, # Путь к папке с файлами для переименования / Path to the folder with files to be renamed
        suffix=None # Суффикс, который надо добавить к именам файлов / Suffix to add to all filenames
        ) -> None:

    '''
        Добавить к имени файла суффикс,
        указанный в параметре suffix
        (или добавить '-1', есть параметр suffix не указан)

        Add suffix to file names 
        (or add '-1' if suffix param not set)
    '''

    # Проходим по всем файлам в папке
    for filename in os.listdir(folder_path):
        old_file_path = os.path.join(folder_path, filename)
        
        # Проверяем, что это файл, а не папка
        if os.path.isfile(old_file_path):
            # Разделяем имя файла и расширение
            name, ext = os.path.splitext(filename)
            # Создаем новое имя файла
            if suffix:
                new_name = f"{name}{suffix}{ext}"
            else:
                new_name = f"{name}-1{ext}"
            new_file_path = os.path.join(folder_path, new_name)
            
            # Переименовываем файл
            os.rename(old_file_path, new_file_path)

    print("Имена файлов успешно изменены.")


def convert_dcm_to_jpg(
        src_path: str, # Пусть к файлу dicom (.dcm) / Path to .dcm file
        dest_path: str, # Путь для сохранения файла .jpg / Path to save .jpg file
        preproc: bool=False # Нужна ли предобработка / if preprocessing needed
        ) -> None:

    '''
        Конвертировать изображение dicom (.dcm) 
        в формат .jpg с нормализацией

        Convert dcm image to .jpg format
        with normalization
    '''
        
    try:
        if preproc:
            image = prepare_dicom_image(src_path, method='clahe')
        else:
            ds = dicom.dcmread(src_path)
            image = ds.pixel_array
        # Нормализация
        if image.dtype != np.uint8:

            window_center, window_width = get_dicom_window_attributes(image)
            min_val = window_center - window_width // 2
            max_val = window_center + window_width // 2

            if max_val != min_val:
                image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)

        img = Image.fromarray(image)
        img.save(dest_path, 'JPEG')
        print(f"Конвертировано {src_path} => {dest_path}")
    except Exception as e:
        print(f"Ошибка при обработке {src_path}: {e}")


def prepare_and_copy_dicom_train_test(
        src_base_dir: str, # Папка с данными / Folder containing data
        dest_base_dir: str # Папка, куда нужно скопировать данные / Folder to copy data
        ) -> None:

    '''
        Предобработать и копировать как train, так и test файлы
        из исходной директории в целевую

        Preprocess and copy both train and test files 
        from source to destination directory
    '''
        
    # Создаем целевые папки
    os.makedirs(os.path.join(dest_base_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_base_dir, 'test'), exist_ok=True)

    for dataset_type in ['train', 'test']:

        src_dir = os.path.join(src_base_dir, dataset_type)
        dest_dir = os.path.join(dest_base_dir, dataset_type)

        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    src_path = os.path.join(root, file)
                    filename_without_ext = os.path.splitext(file)[0]
                    dest_path = os.path.join(dest_dir, f"{filename_without_ext}.jpg")
                    convert_dcm_to_jpg(src_path, dest_path, preproc=True)


def copy_dicom_train_test(
        src_base_dir: str, # Папка с данными / Folder containing data
        dest_base_dir: str # Папка, куда нужно скопировать данные / Folder to copy data
        ) -> None:
    
    '''
        Копировать как train, так и test файлы
        из исходной директории в целевую

        Copy both train and test files 
        from source to destination directory
    '''

    os.makedirs(os.path.join(dest_base_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_base_dir, 'test'), exist_ok=True)

    # Обработка для 'train' и 'test'
    for dataset_type in ['train', 'test']:

        src_dir = os.path.join(src_base_dir, dataset_type)
        dest_dir = os.path.join(dest_base_dir, dataset_type)

        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    src_path = os.path.join(root, file)
                    filename_without_ext = os.path.splitext(file)[0]
                    dest_path = os.path.join(dest_dir, f"{filename_without_ext}.jpg")
                    convert_dcm_to_jpg(src_path, dest_path)


def prepare_and_copy_dicom_test(
        src_base_dir: str, # Папка с данными / Folder containing data
        dest_base_dir: str # Папка, куда нужно скопировать данные / Folder to copy data
        ) -> None:
    
    '''
        Предобработать и копировать test файлы
        из исходной директории в целевую

        Preprocess and copy test files
    '''
    os.makedirs(os.path.join(dest_base_dir, 'test'), exist_ok=True)

    # Обработка для 'train' и 'test'
    for dataset_type in ['test']:
    # for dataset_type in ['train']:

        src_dir = os.path.join(src_base_dir, dataset_type)
        dest_dir = os.path.join(dest_base_dir, dataset_type)

        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    src_path = os.path.join(root, file)
                    filename_without_ext = os.path.splitext(file)[0]
                    dest_path = os.path.join(dest_dir, f"{filename_without_ext}.jpg")
                    convert_dcm_to_jpg(src_path, dest_path, preproc=True)


def copy_dicom_test(
        src_base_dir: str, # Папка с данными / Folder containing data
        dest_base_dir: str # Папка, куда нужно скопировать данные / Folder to copy data
        ) -> None:
    
    '''
        Копировать test файлы
        из исходной директории в целевую

        Copy test files without preprocessing
    '''
        
    # Создаем целевые папки
    os.makedirs(os.path.join(dest_base_dir, 'test'), exist_ok=True)

    # Обработка для 'test'
    for dataset_type in ['test']:

        src_dir = os.path.join(src_base_dir, dataset_type)
        dest_dir = os.path.join(dest_base_dir, dataset_type)

        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    src_path = os.path.join(root, file)
                    filename_without_ext = os.path.splitext(file)[0]
                    dest_path = os.path.join(dest_dir, f"{filename_without_ext}.jpg")
                    convert_dcm_to_jpg(src_path, dest_path)


def prepare_and_copy_dicom(
        src_base_dir: str, # Папка с данными / Folder containing data
        dest_base_dir: str # Папка, куда нужно скопировать данные / Folder to copy data
        ) -> None:
    
    '''
        Предобработать и копировать файлы
        из исходной директории в целевую

        Preprocess and copy files
    '''
    os.makedirs(dest_base_dir, exist_ok=True)

    for root, dirs, files in os.walk(src_base_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                src_path = os.path.join(root, file)
                filename_without_ext = os.path.splitext(file)[0]
                dest_path = os.path.join(dest_base_dir, f"{filename_without_ext}.jpg")
                convert_dcm_to_jpg(src_path, dest_path, preproc=True)


def copy_dicom(
        src_base_dir: str, # Папка с данными / Folder containing data
        dest_base_dir: str # Папка, куда нужно скопировать данные / Folder to copy data
        ) -> None:
    
    '''
        Копировать файлы
        из исходной директории в целевую

        Copy files without preprocessing
    '''
    os.makedirs(dest_base_dir, exist_ok=True)

    for root, dirs, files in os.walk(src_base_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                src_path = os.path.join(root, file)
                filename_without_ext = os.path.splitext(file)[0]
                dest_path = os.path.join(dest_base_dir, f"{filename_without_ext}.jpg")
                convert_dcm_to_jpg(src_path, dest_path)


def debug_files(
        folder_path: str # Путь к папке для проверки / Path to folder for check
        ) -> None:

    '''
        Проверяет консистентность директории,
        чтобы для каждого изображения существовал файл разметки
        
        Check consistensy of directory,
        so that every image has it's own label .txt file
    '''

    all_files = os.listdir(folder_path)
    
    # Создаем словарь для каждого имени файла
    file_dict = {}
    
    for file in all_files:
        name, ext = os.path.splitext(file)
        if name not in file_dict:
            file_dict[name] = {'jpg': False, 'txt': False}
        
        if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            file_dict[name]['jpg'] = True
        elif ext.lower() == '.txt':
            file_dict[name]['txt'] = True
    
    print("Детальный анализ каждого файла:")
    print("=" * 50)
    
    for name, files in sorted(file_dict.items()):
        status = "✅ ПАРА" if files['jpg'] and files['txt'] else "❌ ЛИШНИЙ"
        jpg_status = "🖼️" if files['jpg'] else "  "
        txt_status = "📄" if files['txt'] else "  "
        
        print(f"{status} | {jpg_status} {txt_status} {name}")
    
    # Считаем
    pairs = sum(1 for files in file_dict.values() if files['jpg'] and files['txt'])
    extra = len(file_dict) - pairs
    
    print(f"\n=== РЕЗУЛЬТАТ ===")
    print(f"Пар файлов: {pairs}")
    print(f"Лишних имен: {extra}")
    print(f"Всего файлов должно быть: {pairs * 2}")
    print(f"Фактически файлов: {len(all_files)}")


def merge_txt_files(
        source_dir:str # Директория с выводом модели
):
    # Создаем директорию lib, если она не существует
    lib_dir = Path("./merged")
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

if __name__ == "__main__":
    # Light demo with test file
    pass
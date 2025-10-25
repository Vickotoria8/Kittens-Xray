'''
    Methods used to preprocess images before
    starting the model in train or predict mode
'''

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom as dicom
from pathlib import Path
import numpy as np
import torchxrayvision as xrv

from augmentation import clahe_window, dcp_window, combined_window

def show_pic_with_cv2(
        dcm_sample_image: np.ndarray, # Изображение / Image
        label='DICOM Image' # Название для окна изображения / Name for image window
        ) -> None:
    
    '''
        Отобразить изображение

        Show image
    '''

    max_size = 800
    height, width = dcm_sample_image.shape[:2]
    scale = max_size / max(height, width)
    new_size = (int(width * scale), int(height * scale))
    resized_img = cv2.resize(dcm_sample_image, new_size, interpolation=cv2.INTER_AREA)
    plt.figure(figsize=(10, 8))
    plt.imshow(resized_img, cmap='gray')
    plt.title(label, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def get_dicom_window_attributes(
        ds # Считанное изображение dicom / Dicom image
        ) -> tuple:

    '''
        Вернуть атрибуты WindowCenter и WindowWidth
        для изображения в виде кортежа

        Return WindowCenter and WindowWidth attributes
        of the image as a tuple
    '''

    center = getattr(ds, 'WindowCenter', None)
    if center is None:
        window_center = np.mean(ds.pixel_array)
    elif isinstance(center, dicom.multival.MultiValue):
        window_center = int(center[0]) if 'WindowCenter' in ds else np.mean(ds.pixel_array)
    else:
        window_center = int(center) if 'WindowCenter' in ds else np.mean(ds.pixel_array)

    width = getattr(ds, 'WindowWidth', None)
    if width is None:
        window_width = np.max(ds.pixel_array) - np.min(ds.pixel_array)
    elif isinstance(ds.WindowWidth, dicom.multival.MultiValue):
        window_width = int(ds.WindowWidth[0]) if 'WindowWidth' in ds else np.max(ds.pixel_array) - np.min(ds.pixel_array)
    else:
        window_width = int(ds.WindowWidth) if 'WindowWidth' in ds else np.max(ds.pixel_array) - np.min(ds.pixel_array)
    
    return window_center, window_width


def apply_window_level(
        image: np.ndarray, # Изображение / Image
        window_center, # Параметр WindowCenter изображения dicom (.dcm) / WindowCenter parameter of dicom (.dcm) image
        window_width, # Параметр WindowWidth изображения dicom (.dcm) / WindowWidth parameter of dicom (.dcm) image
        photometric,  # Параметр PhotometricInterpretationя изображени dicom (.dcm) / PhotometricInterpretation parameter of dicom (.dcm) image
        method='simple', # Метод предобработки: simple, CLAHE, DCP или combined / Preprocessing method: simple, CLAHE, SCP or combined
        cL=5.0, # Параметр clicklimit для фильтра clahe / Clicklimit parameter for clahe filter
        tile=(8,8), # Параметр tileGridSize для фильтра clahe / TileGridSize parameter for clahe filter
        patch=15, # Параметр размерности для фильтра DCP / Size parameter for DCP filter
        mode='gray' # Тип фильтра: gray или rgb / Filter type: gray or rgb
        ) -> np.ndarray:
    
    '''
        Нормализовать изображение по оконным параметрам,
        применить фильтры,
        вернуть изображение как numpy.ndarray

        Normalize image with window attributes
        apply filters,
        return image as numpy.ndarray    
    '''
        
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed = image.copy()
    windowed[windowed < img_min] = img_min
    windowed[windowed > img_max] = img_max
    # Нормализация к 0-255 для отображения
    windowed = ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    if photometric == "MONOCHROME1":
        windowed = cv2.bitwise_not(windowed)

    if method == 'clahe':
        windowed = clahe_window(windowed,cL,tile,mode)
    elif method == 'clahe':
        windowed = clahe_window(windowed,cL,tile,mode)
    elif method == 'dcp':
        windowed = dcp_window(windowed,patch)
    elif method == 'combined':
        windowed = combined_window(windowed, cL, patch, tile, mode)
    else:
      pass

    return windowed


def prepare_dicom_image(
        image_path: str, # Путь к изображению dicom (.dcm) / Dicom image path
        method: str='simple', # Метод предобработки: simple, CLAHE, DCP или combined / Preprocessing method: simple, CLAHE, SCP or combined
        cL: float=5.0, # Параметр clicklimit для фильтра clahe / Clicklimit parameter for clahe filter
        tile=(8,8), # Параметр tileGridSize для фильтра clahe / TileGridSize parameter for clahe filter
        patch: int=5, # Параметр размерности для фильтра DCP / Size parameter for DCP filter
        mode: str="gray" # Тип фильтра: gray или rgb / Filter type: gray or rgb
        ) -> np.ndarray:
    
    '''
        Предобработать изображение dicom (.dcm)
        для передачи в модель,
        вернуть изображение как numpy.ndarray

        Preprocess dicom (.dcm) image
        before starting the model,
        return image as numpy.ndarray    
    '''
    # Читаем .dcm
    ds=dicom.dcmread(image_path)
    # Переводим в ndarray для обработки с помощью cv2
    dcm_sample = ds.pixel_array.astype(np.float32)

    window_center, window_width = get_dicom_window_attributes(ds)
    dcm_sample = apply_window_level(dcm_sample, window_center, window_width, ds.PhotometricInterpretation, method, cL, tile, patch, mode)

    return dcm_sample

def get_data_from_dir(
        dir_name: str # Название директории / Directory name
        ) -> list[np.ndarray]:
    
    '''
        Рекурсивно обойти директорию
        для препроцессинга всех dicom (.dcm) файлов

        Go recursively through the directory
        to preprocess all the dicom files
    '''
    
    base_dir = Path(dir_name)
    images = []

    # Собираем необработанные изображения
    for image_path in base_dir.rglob('*'):
        if image_path.is_file() and image_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.dcm'}:
            rel_path = image_path.relative_to(Path('.'))
            images.append(prepare_dicom_image(str(rel_path)))

    # Настраиваем контрастность с помощью аугментации
    for image_path in base_dir.rglob('*'):
        if image_path.is_file() and image_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.dcm'}:
            rel_path = image_path.relative_to(Path('.'))
            images.append(prepare_dicom_image(str(rel_path), method='clahe'))

    return images

if __name__ == "__main__":
    # Light demo with test file
    pass
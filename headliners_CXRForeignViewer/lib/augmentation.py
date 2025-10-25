'''
    Методы аугментации, использованные для
    подготовки train и test датасетов перед
    запуском модели в режиме обучения

    Methods of augmentation used to prepare
    train and test images before starting 
    the model in train mode
'''
import numpy as np
import cv2

def clahe_window(
        windowed: np.ndarray, # Нормализованное по оконным параметрам изображение / Image normalized with window  attributes
        cL: float, # Параметр clicklimit для фильтра clahe / Clicklimit parameter for clahe filter
        tile, # Параметр tileGridSize для фильтра clahe / TileGridSize parameter for clahe filter
        mode: str="gray" # Тип фильтра: gray или rgb / Filter type: gray or rgb
        ) -> np.ndarray:
    
    '''
        Применить фильтр CLAHE на изображении,
        вернуть изображение в формате numpy.adarray

        Apply CLAHE filter to the image, 
        return image as numpy.ndarray 
    '''

    if mode == "gray":
      clahe = cv2.createCLAHE(clipLimit=cL, tileGridSize=tile)
      windowed = clahe.apply(windowed)
    else:
      if len(windowed.shape) == 2:
          windowed_rgb = cv2.cvtColor(windowed, cv2.COLOR_GRAY2BGR)
      else:
          windowed_rgb = windowed.copy()

      clahe = cv2.createCLAHE(clipLimit=cL, tileGridSize=tile)
      lab = cv2.cvtColor(windowed_rgb, cv2.COLOR_BGR2LAB)
      l, a, b = cv2.split(lab)  # разделяем на каналы
      l2 = clahe.apply(l)
      lab = cv2.merge((l2, a, b))
      windowed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
      windowed = cv2.cvtColor(windowed, cv2.COLOR_BGR2GRAY)

    return windowed


def dcp_window(
        windowed: np.ndarray, # Нормализованное по оконным параметрам изображение / Image normalized with window  attributes
        patch_size: int # Параметр размерности для фильтра DCP / Size parameter for DCP filter
        ) -> np.ndarray:

    '''
        Применить фильтр DCP на изображении,
        вернуть изображение в формате numpy.adarray

        Apply DCP filter to the image, 
        return image as numpy.ndarray 
    '''
        
    if len(windowed.shape) == 2:
          windowed_rgb = cv2.cvtColor(windowed, cv2.COLOR_GRAY2BGR)
    else:
          windowed_rgb = windowed.copy()

    b, g, r = cv2.split(windowed_rgb)
    min_bg = cv2.min(b, g)
    dark_channel = cv2.min(min_bg, r)
    kernel = np.ones((patch_size, patch_size), np.uint8)
    dark_channel = cv2.erode(dark_channel, kernel, iterations=1)

    return dark_channel


def combined_window(
        windowed_rgb: np.ndarray, # Нормализованное по оконным параметрам изображение в формате RGB / RGB image normalized with window attributes
        cL: float, # Параметр clicklimit для фильтра clahe / Clicklimit parameter for clahe filter
        tile, # Параметр tileGridSize для фильтра clahe / TileGridSize parameter for clahe filter
        patch: int, # Параметр размерности для фильтра DCP / Size parameter for DCP filter
        mode: str="gray" # Тип фильтра: gray или rgb / Filter type: gray or rgb
        ):

    '''
        Применить комбинированный фильтр (CLAHE + DCP) на изображении,
        вернуть изображение в формате numpy.adarray

        Apply combines filter (CLAHE + DCP) to the image, 
        return image as numpy.ndarray 
    '''
        
    if mode == "rgb":
      clahed = clahe_window(windowed_rgb,cL,tile,mode)
    elif mode == "gray":
      clahed = clahe_window(windowed_rgb,cL,tile,mode)
      clahed = cv2.cvtColor(windowed_rgb, cv2.COLOR_GRAY2BGR)
    final = dcp_window(clahed, patch)

    return final

if __name__ == "__main__":
    # Light demo with test file
    pass
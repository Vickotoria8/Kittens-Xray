import os
import random
import shutil
from tqdm.notebook import tqdm

from ultralytics.models import YOLO
from .. import config

from .file_management import prepare_and_copy_dicom, copy_dicom, cleanup_directories
from .postprocess import merge_txt_files, draw_boxes, make_binary

class YoloViewer:

    '''
        Mapper class for convenient use of model in Python code
    '''

    def __init__(
            self, 
            trained = True, # if pretrained model is used
            model_path: str = config.MODEL_PATH # weights fore model
            ):

        self.trained = trained
        self.model_path = model_path
        
        # Metrics

    def _train_test_split(self, path, neg_path=None, split=0.2):
        print("------ PROCESS STARTED -------")

        files = list(set([name[:-4] for name in os.listdir(path)]))
        random.seed(42)
        random.shuffle(files)

        test_size = int(len(files) * split)

        # РАЗДЕЛЕНИЕ на непересекающиеся части
        train_files = files[:-test_size]  # 80% для тренировки
        val_files = files[-test_size:]    # 20% для валидации

        # Создание директорий
        os.makedirs(config.train_path_img, exist_ok=True)
        os.makedirs(config.train_path_label, exist_ok=True)
        os.makedirs(config.val_path_img, exist_ok=True)
        os.makedirs(config.val_path_label, exist_ok=True)

        # Копирование ТОЛЬКО тренировочных данных
        for filex in tqdm(train_files):
            if filex == 'classes':
                continue
            shutil.copy2(path + filex + '.jpg', f"{config.train_path_img}/" + filex + '.jpg')
            shutil.copy2(path + filex + '.txt', f"{config.train_path_label}/" + filex + '.txt')

        print(f"------ Training data created with {len(train_files)} images -------")

        # Копирование ТОЛЬКО валидационных данных
        for filex in tqdm(val_files):
            if filex == 'classes':
                continue
            shutil.copy2(path + filex + '.jpg', f"{config.val_path_img}/" + filex + '.jpg')
            shutil.copy2(path + filex + '.txt', f"{config.val_path_label}/" + filex + '.txt')

        print(f"------ Validation data created with {len(val_files)} images ----------")
        print("------ TASK COMPLETED -------")

    def train(
            self, 
            project_name: str=config.PROJECT, # directory for model weights and params #TODO: set default value
            model_name: str=config.MODEL_NAME,
            path_to_yaml=config.PATH_TO_YAML,
            ) -> None:
        
        self._train_test_split(config.TRAIN_FILES_PATH)

        model = YOLO('yolov8s.pt')

        results = model.train(
            data=path_to_yaml,
            epochs=200,
            imgsz=640,
            batch=8,
            project=project_name,
            name=model_name,
            exist_ok=True,
        )

        self.model_path = 'my_model/model_data/weights/best.onnx'

    def predict(
            self, 
            test_data, # data for prediction
            output_directory, # directory to save output images #TODO: set default value
        ) -> None:

        prepare_and_copy_dicom(test_data, config.TEMP_TEST_DATA_PATH)
        copy_dicom(test_data, config.TEMP_TEST_DATA_PATH)

        model = YOLO(self.model_path)

        results = model.predict(
            task='detect',
            mode='predict',
            source=config.TEMP_TEST_DATA_PATH,
            conf=0.25,
            save_txt=True
        )

        merge_txt_files(config.OUTPUT_LABELS_PATH)
        draw_boxes()
        make_binary()
        cleanup_directories(['runs', config.TEMP_TEST_DATA_PATH])
from ultralytics.models import YOLO

# requirements.txt

class YoloViewer:

    '''
        Mapper class for convenient use of model in Python code
    '''

    def __init__(
            self, 
            trained = False, # if pretrained model is used
            calculated_weights = None # weights fore model
            ):

        self.trained = trained
        if self.trained:
            self.weights = 'path/to/our/weights'
        else:
            self.weights = calculated_weights
        # Metrics

    def train(
            self, 
            train_data, # data for training
            model_directory, # directory for model weights and params #TODO: set default value
            **kwargs # Model parameters
            ) -> None:
        
        # training logic
        # path_to_weights = path/to/weights
        # self.weights = path_to_weights

        model = YOLO('yolov8s.pt')

        results = model.train(
            data='/content/drive/MyDrive/shuffle_train_invert_clean_new/dataset_task3.yaml',
            epochs=200,
            imgsz=640,
            batch=8,
            project='/content/drive/MyDrive/shuffle_train_invert_clean_new/yolo_training_results',
            name='foreign_items_task10',
            exist_ok=True
        )
        pass

    def predict(
            self, 
            test_data, # data for prediction
            output_directory, # directory to save output images #TODO: set default value
            **kwargs # Model parameters
        ) -> None:

        # prediction logic using class's weights
        pass
from ..lib.model_mapper import YoloViewer

def main():
    data = input("Введите путь к папке с файлами DICOM: ")

    if not data or len(data) == 0:
        data = 'headliners_CXRForeignViewer/demo_files'
    print("Target data directory: ", data)

    model = YoloViewer(trained=True)
    model.predict(data)
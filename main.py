import os
import json
import dotenv 
from scripts import ResizeImages, SplitDataset, ConvertH5ToTFLite
from train_mobilenetv2 import TrainModelMobilenetV2
from evaluate_batch import EvaluateModel


if __name__ == "__main__":
    # Cargar variables de entorno
    dotenv.load_dotenv()
    
    # Configuración
    IMAGE_SIZE = tuple(json.loads(os.getenv('IMAGE_SIZE')))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
    EPOCHS = int(os.getenv('EPOCHS'))
    BASE_DIR = os.getenv('DIR_DATA_BASE')
    LABELS = json.loads(os.getenv('LABELS'))
    MODEL_SAVE_PATH = os.getenv('MODEL_PATH_H5')
    USED_MODEL = os.getenv('MODEL_PATH_TFLITE')
    
    #Redimensionar imágenes
    input_path = os.path.join(BASE_DIR, "original")
    output_path = os.path.join(BASE_DIR,"procesados")
    resize_images = ResizeImages(input_path, output_path, IMAGE_SIZE)
    resize_images.ejecutar()
    
    #Dividir el dataset
    source_root_dir = os.path.join(BASE_DIR, "procesados")
    dest_root_dir = os.path.join(BASE_DIR, "procesados_split")
    split_dataset = SplitDataset(source_root_dir, dest_root_dir, LABELS, test_split_ratio=0.2, random_seed=42)
    split_dataset.ejecutar()

    # Entrenar el modelo
    trainer = TrainModelMobilenetV2(IMAGE_SIZE, BATCH_SIZE, EPOCHS, BASE_DIR, LABELS, MODEL_SAVE_PATH)
    trainer.ejecutar()

    # Convertir el modelo a TFLite
    converter = ConvertH5ToTFLite(MODEL_SAVE_PATH, os.getenv('MODEL_PATH_TFLITE'))
    converter.run()

    # Evaluar el modelo
    BASE_DIR = os.path.join(BASE_DIR, "procesados_split", "test")
    evaluate_batch = EvaluateModel(USED_MODEL, BASE_DIR, LABELS, IMAGE_SIZE)
    evaluate_batch.evaluate()
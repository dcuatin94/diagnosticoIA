import os
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report

IMAGE_SIZE = (150, 150)
MODEL_PATH = "model.tflite"
DATA_DIR = "data/procesados"
BATCH_SIZE = 32

def load_images_and_labels():
    images = []
    labels = []
    label_map = {"COVID": 0, "NORMAL": 1, "VIRAL_PNEUMONIA": 2}

    for label_name, label_index in label_map.items():
        class_dir = Path(DATA_DIR) / label_name
        for img_file in class_dir.glob("*.jpg"):
            image = cv2.imread(str(img_file))
            image = cv2.resize(image, IMAGE_SIZE)
            image = image.astype(np.float32) / 255.0
            images.append(image)
            labels.append(label_index)
    return np.array(images), np.array(labels)

def predict_tflite(interpreter, input_data):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    predictions = []
    for i in range(0, len(input_data), BATCH_SIZE):
        batch = input_data[i:i+BATCH_SIZE]
        for img in batch:
            interpreter.set_tensor(input_index, np.expand_dims(img, axis=0))
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)
            predictions.append(np.argmax(output))
    return predictions

if __name__ == "__main__":
    print("📦 Cargando imágenes...")
    X, y_true = load_images_and_labels()

    print("🔍 Cargando modelo TFLite...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    print("🤖 Realizando predicciones...")
    y_pred = predict_tflite(interpreter, X)

    print("\n📊 Resultados:")
    print(classification_report(y_true, y_pred, target_names=["COVID", "NORMAL", "VIRAL_PNEUMONIA"]))

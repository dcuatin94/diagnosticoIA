import os
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report

image_size = (150, 150)
model_path = "mobilenetv2_model.tflite"

# Directorio donde están las imágenes procesadas
data_dir = "datos/procesados"
batch_size = 32

def load_images_and_labels():
    images = []
    labels = []
    label_map = {"COVID": 0, "Lung_Opacity": 1, "Normal": 2, "Viral_Pneumonia": 3}

    for label_name, label_index in label_map.items():
        class_dir = Path(data_dir) / label_name
        for img_file in class_dir.glob("*.jpg"):
            image = cv2.imread(str(img_file))
            image = cv2.resize(image, image_size)
            image = image.astype(np.float32) / 255.0
            images.append(image)
            labels.append(label_index)
        for img_file in class_dir.glob("*.png"):
            image = cv2.imread(str(img_file))
            image = cv2.resize(image, image_size)
            image = image.astype(np.float32) / 255.0
            images.append(image)
            labels.append(label_index)
    return np.array(images), np.array(labels)

def predict_tflite(interpreter, input_data):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    predictions = []
    for i in range(0, len(input_data), batch_size):
        batch = input_data[i:i+batch_size]
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
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    print("🤖 Realizando predicciones...")
    y_pred = predict_tflite(interpreter, X)

    print("\n📊 Resultados:")
    print(classification_report(y_true, y_pred, target_names=["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]))

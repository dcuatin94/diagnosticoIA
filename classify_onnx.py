# classify_onnx.py

import onnxruntime as ort # Importar ONNX Runtime
import numpy as np
import cv2
import os
import time

# --- CONFIGURACIÓN ---
ONNX_MODEL_PATH = 'models/pulmonar_classifier.onnx' # Ruta a tu modelo ONNX
IMG_HEIGHT, IMG_WIDTH = 150, 150
CLASS_NAMES = ['COVID', 'Normal', 'Viral_Pneumonia'] # Asegúrate que coincida con tu orden de entrenamiento

def preprocess_image_for_inference(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Carga una imagen, la redimensiona y la normaliza para el modelo ONNX.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen {image_path}.")
        return None

    # Asegurarse de que la imagen tenga 3 canales (RGB)
    if len(img.shape) == 2: # Si es escala de grises
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4: # Si tiene canal alfa (RGBA)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    # Normalizar a [0, 1] y convertir a float32
    input_data = resized_img.astype(np.float32) / 255.0
    # Agregar dimensión de batch (1, H, W, C)
    input_data = np.expand_dims(input_data, axis=0)

    return input_data

def main():
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"Error: Modelo ONNX no encontrado en '{ONNX_MODEL_PATH}'.")
        print("Asegúrate de haber convertido el modelo Keras a ONNX primero.")
        return

    # Crear una sesión de inferencia ONNX Runtime
    # Se puede especificar 'CPUExecutionProvider' para asegurar que corre en CPU
    try:
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Error al cargar el modelo ONNX con ONNX Runtime: {e}")
        print("Asegúrate de que el modelo ONNX fue convertido correctamente y onnxruntime está instalado.")
        return

    # Obtener los nombres de las entradas y salidas del modelo ONNX
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Ejemplo de uso: clasificar una imagen de prueba
    # Reemplaza con la ruta a una imagen de prueba real
    test_image_path = 'datos/procesados/COVID/images/COVID-1.png' # ¡Cambia esto a una imagen real!

    if not os.path.exists(test_image_path):
        print(f"Error: Imagen de prueba no encontrada en '{test_image_path}'.")
        print("Por favor, proporciona una ruta válida a una imagen de prueba.")
        return

    print(f"Clasificando imagen con ONNX Runtime: {test_image_path}")
    input_data = preprocess_image_for_inference(test_image_path)

    if input_data is None:
        return

    # Realizar inferencia
    start_time = time.time()
    # input_feed es un diccionario {nombre_de_entrada: datos_de_entrada}
    outputs = session.run([output_name], {input_name: input_data})
    end_time = time.time()

    prediction_probabilities = outputs[0][0] # El resultado es una lista de arrays, y luego un array de batch
    prediction_index = np.argmax(prediction_probabilities)

    print(f"Predicción: {CLASS_NAMES[prediction_index]}")
    print(f"Probabilidades: {prediction_probabilities}")
    print(f"Tiempo de inferencia para una imagen con ONNX Runtime: {end_time - start_time:.4f} segundos")

if __name__ == "__main__":
    main()
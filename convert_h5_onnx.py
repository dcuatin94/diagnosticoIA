import tensorflow as tf
import tf2onnx
import onnx
import os
import sys

# --- CONFIGURACIÓN ---
KERAS_MODEL_PATH = 'models/mobilenetv2_model.h5' # Ruta a tu modelo .h5 entrenado
ONNX_MODEL_SAVE_PATH = 'models/pulmonar_classifier.onnx' # Ruta donde se guardará el modelo ONNX

def convert_keras_to_onnx(keras_model_path, onnx_model_path):
    """
    Carga un modelo Keras (.h5) y lo convierte a formato ONNX.
    """
    if not os.path.exists(keras_model_path):
        print(f"Error: El modelo Keras no se encontró en '{keras_model_path}'.")
        print("Asegúrate de que la ruta sea correcta y el modelo ha sido entrenado y guardado.")
        sys.exit(1) # Salir con un código de error

    print(f"Cargando modelo Keras desde: {keras_model_path}")
    try:
        model = tf.keras.models.load_model(keras_model_path)
    except Exception as e:
        print(f"Error al cargar el modelo Keras: {e}")
        print("Asegúrate de que TensorFlow esté correctamente instalado y el archivo .h5 no esté corrupto.")
        sys.exit(1)

    print("Modelo Keras cargado exitosamente. Iniciando conversión a ONNX...")

    # Define la forma de la entrada para la conversión
    # Asume que tu modelo MobileNetV2 fue entrenado con (150, 150, 3)
    # y espera un batch, por lo que la forma completa es (None, 150, 150, 3)
    input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="input")]

    try:
        # Convertir el modelo Keras a ONNX
        # opset=13 es una versión común y compatible para muchos runtimes ONNX
        onnx_model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=13,
            output_path=onnx_model_path # Escribe directamente al archivo
        )
        print(f"✅ Modelo ONNX guardado exitosamente en: {onnx_model_path}")

        # Opcional: Verificar el modelo ONNX para asegurar su validez
        # onnx.checker.check_model(onnx_model_proto)
        # print("Verificación del modelo ONNX exitosa.")

    except Exception as e:
        print(f"Error durante la conversión a ONNX: {e}")
        print("Posibles causas:")
        print("  - `tf2onnx` no está instalado (pip install tf2onnx)")
        print("  - Incompatibilidad de operaciones de Keras con ONNX (revisa la documentación de tf2onnx)")
        sys.exit(1)

if __name__ == "__main__":
    # Asegúrate de que la carpeta de modelos exista
    os.makedirs(os.path.dirname(ONNX_MODEL_SAVE_PATH), exist_ok=True)

    convert_keras_to_onnx(KERAS_MODEL_PATH, ONNX_MODEL_SAVE_PATH)
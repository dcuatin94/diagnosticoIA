import tensorflow as tf
import tf2onnx
import onnx
import os
import sys

# Configuración
KERAS_MODEL_PATH = "models/mobilenetv2_model.h5"
ONNX_MODEL_SAVE_PATH = "models/pulmonar_classifier.onnx"

def convert_keras_to_onnx_safe(model_path, output_path):
    if not os.path.exists(model_path):
        print(f"❌ No se encontró el modelo: {model_path}")
        sys.exit(1)

    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Modelo Keras cargado.")

        # Crear una función de inferencia desde el modelo
        spec = (tf.TensorSpec((None, 150, 150, 3), tf.float32, name="input"),)
        model_func = tf.function(model).get_concrete_function(spec)

        print("🔄 Convirtiendo a ONNX (opset 13)...")
        model_proto, _ = tf2onnx.convert.from_function(
            model_func,
            input_signature=spec,
            opset=13,
            output_path=output_path
        )

        onnx.checker.check_model(model_proto)
        print(f"✅ Modelo ONNX guardado y validado en: {output_path}")

    except Exception as e:
        print(f"❌ Error durante la conversión: {e}")
        sys.exit(1)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(ONNX_MODEL_SAVE_PATH), exist_ok=True)
    convert_keras_to_onnx_safe(KERAS_MODEL_PATH, ONNX_MODEL_SAVE_PATH)

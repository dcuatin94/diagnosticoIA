import tensorflow as tf

MODEL_PATH = "models/mobilenetv2_model.h5"  # Ruta al modelo Keras
TFLITE_PATH = "models/model_int8.tflite"

# Cargar modelo
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado.")

# Convertidor
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Cuantización dinámica
tflite_model = converter.convert()

# Guardar archivo
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ Modelo TFLite guardado en: {TFLITE_PATH}")

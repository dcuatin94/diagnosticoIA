import os
import dotenv
import tensorflow as tf

class ConvertH5ToTFLite:
    def __init__(self, model_path, tflite_path):
        self.model_path = model_path
        self.tflite_path = tflite_path

    def load_model(self):
        model = tf.keras.models.load_model(self.model_path)
        print("✅ Modelo cargado.")
        return model

    def convert_to_tflite(self, model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Cuantización dinámica
        tflite_model = converter.convert()
        with open(self.tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"✅ Modelo TFLite guardado en: {self.tflite_path}")

    def run(self):
        model = self.load_model()
        self.convert_to_tflite(model)
        
    # def main():
    #     dotenv.load_dotenv()
    #     model_path = os.getenv('MODEL_PATH_H5')
    #     tflite_path = os.getenv('MODEL_PATH_TFLITE')
    #     if not model_path or not os.path.exists(model_path):
    #         print(f"❌ No se encontró el modelo H5 en: {model_path}")
    #         return
    #     converter = ConvertH5ToTFLite(model_path, tflite_path)
    #     model = converter.load_model()
    #     converter.convert_to_tflite(model)

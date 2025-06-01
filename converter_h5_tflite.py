import tensorflow as tf

model = tf.keras.models.load_model("mobilenetv2_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("mobilenetv2_model.tflite", "wb") as f:
    f.write(tflite_model)
    
    
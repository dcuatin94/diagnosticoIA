import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Configuraciones
image_size = (150, 150)
batch_size = 32
epocas = 10
data_dir = "datos/procesados"
modelo = "mobilenetv2_model.h5"

# 1. Preprocesamiento y Aumento de Datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# 2. Cargar modelo base
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*image_size, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # congelar pesos

# 3. Construir modelo final
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 4. Entrenar
history = model.fit(
    train_generator,
    epochs=epocas,
    validation_data=val_generator
)
# 5. Guardar modelo
model.save(modelo)
print(f"✅ Modelo guardado en: {modelo}")

# 6. Graficar accuracy
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.title('Precisión del modelo')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()

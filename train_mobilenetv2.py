import os
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import matplotlib.pyplot as plt
from custom_data_loader import LungImageGenerator

# Configuración
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 15
BASE_DIR = "datos/procesados_split" 
LABELS = ["COVID", "Normal", "Viral_Pneumonia"]
MODEL_SAVE_PATH = "models/mobilenetv2_model.h5"
os.makedirs("models", exist_ok=True)
# Generadores
train_gen = LungImageGenerator(os.path.join(BASE_DIR, "train"), LABELS, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True, augment=True)
val_gen = LungImageGenerator(os.path.join(BASE_DIR, "test"), LABELS, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False, augment=False)

if len(train_gen) == 0:
    print(f"Error: El generador de entrenamiento no contiene datos. Verifique la ruta '{os.path.join(BASE_DIR, 'train')}' y los nombres de las clases.")
    exit()
if len(val_gen) == 0:
    print(f"Error: El generador de validación/prueba no contiene datos. Verifique la ruta '{os.path.join(BASE_DIR, 'test')}' y los nombres de las clases.")
    exit()


# Modelo base
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # congelar

# Modelo completo
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(len(LABELS), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# Guardar modelo
model.save(MODEL_SAVE_PATH)
print(f"✅ Modelo guardado en: {MODEL_SAVE_PATH}")

# Gráfico
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.legend()
plt.title("Precisión del modelo con MobileNetV2")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig("training_accuracy.png")
plt.show()

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models # type: ignore
from utils import LungImageGenerator

class TrainModelMobilenetV2:
    def __init__(self, image_size, batch_size, epochs, base_dir, labels, model_save_path):
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.base_dir = os.path.join(base_dir, "procesados_split")
        self.labels = labels
        self.model_save_path = model_save_path
    
    def validar_directorio(self):
        if not os.path.isdir(self.base_dir):
            raise FileNotFoundError(f"El directorio base '{self.base_dir}' no existe. Por favor, verifica la ruta.")
        print(f"✅ Directorio base válido: {self.base_dir}")
        
        train_get = LungImageGenerator(
            base_dir=os.path.join(self.base_dir, "train"),
            labels=self.labels,
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=True,
            augment=True
        )
        val_gen = LungImageGenerator(
            base_dir=os.path.join(self.base_dir, "test"),
            labels=self.labels,
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=False,
            augment=False
        )
        return train_get, val_gen

    def ejecutar(self):
        train_gen, val_gen = self.validar_directorio()
        model = self.construir_modelo()
        model = self.compilar_model(model)
        history = self.entrenar_modelo(model, train_gen, val_gen)
        self.guardar_modelo(model)
        print(f"✅ Entrenamiento completado. Guardando el modelo en: {self.model_save_path}")
        self.generar_grafico(history)

    def generar_modelo(self):
        # Modelo base
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        return base_model

    def construir_modelo(self):
        base_model = self.generar_modelo()
        # Modelo completo
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(len(self.labels), activation='softmax')
        ])
        return model
    
    def compilar_model(self, model):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def entrenar_modelo(self, model, train_gen, val_gen):
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.epochs
        )
        return history
    
    def guardar_modelo(self, model):
        # Guardar el modelo en formato H5
        model.save(self.model_save_path)
        print(f"✅ Modelo guardado en: {self.model_save_path}")
        
    def generar_grafico(self, history):
        # Gráfico de precisión
        plt.plot(history.history["accuracy"], label="Entrenamiento")
        plt.plot(history.history["val_accuracy"], label="Validación")
        plt.legend()
        plt.title("Precisión del modelo con MobileNetV2")
        plt.xlabel("Épocas")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.savefig("reports/training_accuracy.png")
        plt.show()

# Proyecto Diagnóstico IA - Clasificación de Radiografías de Tórax

Este proyecto implementa un pipeline completo para el preprocesamiento, segmentación, entrenamiento, evaluación y despliegue de modelos de deep learning para el diagnóstico automático de imágenes de tórax (COVID, Lung Opacity, Normal, Viral Pneumonia).

## Estructura del Proyecto

```
diagnosticoIA/
│
├── datos/
│   ├── original/           # Dataset original sin modificar
│   ├── procesados/         # Imágenes y máscaras redimensionadas y preprocesadas
│   └── procesados_split/   # Datos divididos en train/test por clase
│
├── models/                 # Modelos entrenados (.h5, .tflite)
│
├── scripts/                # Scripts para cada etapa del pipeline
│   ├── resize_images.py        # Redimensiona imágenes y aplica máscaras
│   ├── mover_imagenes.py       # Utilidad para mover imágenes
│   ├── split_dataset.py        # Divide el dataset en train/test
│   ├── converter_h5_tflite.py  # Convierte modelos Keras a TFLite
│   └── __init__.py             # Exporta clases para uso en main.py
│
├── utils/
│   └── custom_data_loader.py   # Generador personalizado para cargar imágenes y máscaras
│
├── main.py                 # Pipeline principal (ejecuta todo el flujo)
├── dashboard.py            # Aplicación Streamlit (Interfaz de Usuario)
|── train_mobilenetv2.py    # Entrenamiento del modelo MobileNetV2
├── evaluate_batch.py       # Evaluación por lotes del modelo
├── requirements.txt        # Dependencias del proyecto
└── README.txt              # (Este archivo)
```

## Flujo de Trabajo

1. **Preprocesamiento:**
   - Redimensiona imágenes y aplica máscaras (scripts/resize_images.py).
   - Mueve imágenes si es necesario (scripts/mover_imagenes.py).

2. **División del Dataset:**
   - Divide los datos en conjuntos de entrenamiento y prueba (scripts/split_dataset.py).

3. **Entrenamiento:**
   - Entrena un modelo MobileNetV2 usando generadores personalizados (train_mobilenetv2.py).

4. **Conversión:**
   - Convierte el modelo entrenado a formato TFLite para despliegue eficiente (scripts/converter_h5_tflite.py).

5. **Evaluación:**
   - Evalúa el modelo en lotes y genera métricas de desempeño (evaluate_batch.py).

6. **Despliegue:**
   - Ejecutar `main.py` para ejecutar todo el flujo.
   - Los modelos Keras y TFLite se guardan en la carpeta `models/` 
   - Ejecutar  `streamlit run app_dashboard.py --server.port 8502` para ejecutar la interfaz de usuario.

## Clases y Scripts Principales

- `ResizeImages`: Redimensiona imágenes y aplica máscaras.
- `SplitDataset`: Divide el dataset en train/test y copia imágenes/máscaras.
- `LungImageGenerator`: Generador Keras para cargar imágenes y máscaras con aumentos.
- `TrainModelMobilenetV2`: Clase para entrenar el modelo MobileNetV2.
- `ConvertH5ToTFLite`: Convierte modelos Keras a TFLite.

## Cómo Ejecutar

1. Configura las variables de entorno en un archivo `.env` (ver ejemplos en el código).
2. Ejecuta `main.py` para correr el pipeline completo, o ejecuta los scripts individualmente según la etapa que desees.

## Requisitos

- Python 3.8+
- TensorFlow, OpenCV, Albumentations, scikit-learn, tqdm, dotenv, matplotlib
- Instala las dependencias con `pip install -r requirements.txt`
- Crear un archivo `datos/` con las carpetas `original/` y `procesados/`

## Notas

- La estructura de carpetas de datos debe respetar el formato: `Clase/images/*.png` y `Clase/masks/*.png`.
- El pipeline es modular y puede adaptarse fácilmente a nuevos datasets o modelos.

---
> [!NOTE]
> # Resumen de Desempeño

### Precisión del modelo con MobileNetV2 en entrenamiento:

[![Precisión del modelo con MobileNetV2](/diagnosticoIA/reports/training_accuracy.png)](training_accuracy.png)
## Evaluación del modelo TFLite:
## 📊Clasificación

                 Precision    Recall  F1-score   Support

             COVID       0.84      0.72      0.78       724
            Normal       0.90      0.94      0.92      2039
      Viral Pneumonia    0.88      0.91      0.89       269

       accuracy                           0.88      3032
      macro avg       0.87      0.86      0.86      3032
      weighted avg    0.88      0.88      0.88      3032

- ### ✅ Precisión total: 88.26%

[![Evaluación del modelo](/diagnosticoIA/evaluation_result.png)](evaluation_result.png) 

**Autor:** Daniel Cuatin, Miguel Guevara, Pedro Valverde
**Fecha:** Junio 2025
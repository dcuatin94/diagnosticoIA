import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import os
from PIL import Image
from datetime import datetime

# Configuración
MODEL_PATH = "models/model_int8.tflite"
IMAGE_SIZE = (150, 150)
LABELS = ["COVID", "Normal", "Viral_Pneumonia"]
CSV_PATH = "resultados.csv"

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image, mask=None):
    image = image.resize(IMAGE_SIZE)
    img = np.array(image).astype(np.float32)

    if mask:
        mask = mask.resize(IMAGE_SIZE).convert("L")
        mask_np = np.array(mask) / 255.0
        mask_np = np.expand_dims(mask_np, axis=-1)
        img = img * mask_np
    else:
        lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0)
        cl = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

    img = img / 255.0
    return np.expand_dims(img, axis=0), img.astype(np.float32)

def preprocess_for_tflite(img, input_details):
    if input_details['dtype'] == np.int8:
        scale, zero_point = input_details['quantization']
        img = img / scale + zero_point
        img = np.round(img).astype(np.int8)
    return img

def predict_image(interpreter, image_batch):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_index = input_details["index"]
    output_index = output_details["index"]
    
    input_data = input_data.astype(np.float32)
    interpreter.set_tensor(input_index, input_data)

    # input_data = preprocess_for_tflite(image_batch, input_details)
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)

    if output_details['dtype'] == np.int8:
        scale, zero_point = output_details['quantization']
        output = scale * (output.astype(np.float32) - zero_point)

    return output[0]

def guardar_resultado(nombre_archivo, resultado, probabilidades):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fila = {
        "fecha": now,
        "imagen": nombre_archivo,
        "diagnostico": resultado,
        "prob_COVID": float(probabilidades[0]),
        "prob_NORMAL": float(probabilidades[1]),
        "prob_VIRAL_PNEUMONIA": float(probabilidades[2])
    }

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, pd.DataFrame([fila])], ignore_index=True)
    else:
        df = pd.DataFrame([fila])

    df.to_csv(CSV_PATH, index=False)

# Interfaz de usuario
st.set_page_config(page_title="🩺 Dashboard Diagnóstico Pulmonar", layout="wide")
st.title("🩻 Diagnóstico de Enfermedades Pulmonares con IA (TFLite + Streamlit)")

col1, col2 = st.columns(2)

with col1:
    st.header("📂 Subir Imagen para Diagnóstico")
    uploaded_image = st.file_uploader("Imagen del paciente", type=["jpg", "png"])
    uploaded_mask = st.file_uploader("Máscara segmentada (opcional)", type=["jpg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        mask = Image.open(uploaded_mask).convert("L") if uploaded_mask else None
        st.image(image, caption="🖼️ Imagen original", use_column_width=True)

        with st.spinner("🧠 Procesando con modelo TFLite..."):
            interpreter = load_model()
            image_batch, processed_image = preprocess_image(image, mask)
            prediction = predict_image(interpreter, image_batch)
            predicted_label = LABELS[np.argmax(prediction)]

        st.success(f"📌 Diagnóstico: **{predicted_label}**")
        st.bar_chart({LABELS[i]: float(p) for i, p in enumerate(prediction)})
        st.image(processed_image, caption="🔍 Imagen procesada", use_column_width=True)

        guardar_resultado(
            nombre_archivo=uploaded_image.name,
            resultado=predicted_label,
            probabilidades=prediction
        )
        st.info("💾 Resultado almacenado en `resultados.csv`")

with col2:
    st.header("📊 Resultados & Evaluación")
    
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        st.subheader("📄 Últimos diagnósticos")
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.warning("Aún no se han registrado diagnósticos.")

    if os.path.exists("training_accuracy.png"):
        st.subheader("📈 Precisión durante entrenamiento")
        st.image("training_accuracy.png", caption="Precisión por época", use_column_width=True)

    if os.path.exists("evaluation_result.png"):
        st.subheader("🧪 Evaluación del modelo")
        st.image("evaluation_result.png", caption="Comparación: Etiquetas reales vs predichas", use_column_width=True)

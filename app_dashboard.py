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
CSV_PATH = "resultados.csv
LOAD_NEW_DIR = "datos/load_new"
os.makedirs(LOAD_NEW_DIR, exist_ok=True)
MASK_NEW_DIR = "datos/mask_new"
os.makedirs(MASK_NEW_DIR, exist_ok=True)

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image):
    image = image.resize(IMAGE_SIZE)
    img = np.array(image).astype(np.float32)

    # Mejora de contraste con CLAHE
    lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    cl = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

    img = img / 255.0
    img = img.astype(np.float32)
    return np.expand_dims(img, axis=0), img

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
    input_data = preprocess_for_tflite(image_batch, input_details)

    
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


def generar_mascara_binaria(image_np, output_path):
    # Convertir a escala de grises
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Mejorar contraste con CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0)
    gray = clahe.apply(gray)

    # Aplicar umbral adaptativo
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   blockSize=31, C=5)

    # Aplicar operaciones morfológicas para limpiar ruido
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)

    # Mantener solo las regiones más grandes (pulmones)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # filtrar regiones pequeñas
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    cv2.imwrite(output_path, mask)

# Interfaz de usuario
st.set_page_config(page_title="🩺 Dashboard Diagnóstico Pulmonar", layout="wide")
st.title("🩻 Diagnóstico de Enfermedades Pulmonares con IA (TFLite + Streamlit)")

col1, col2 = st.columns(2)

with col1:
    st.header("📂 Subir Imagen para Diagnóstico")
    uploaded_image = st.file_uploader("Imagen del paciente", type=["jpg", "png"])

    if uploaded_image:
        image_filename = uploaded_image.name
        image_path = os.path.join(LOAD_NEW_DIR, image_filename)

        if os.path.exists(image_path):
            st.warning("⚠️ Esta imagen ya ha sido procesada previamente.")
        else:
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            st.success(f"✅ Imagen almacenada en 'load_new/'")
            
            # Generar máscara
            image_np = np.array(Image.open(image_path).convert("RGB"))
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

            # Opcional: rellenar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

            # Guardar máscara en mask_new
            mask_path = os.path.join(MASK_NEW_DIR, image_filename)
            cv2.imwrite(mask_path, mask)
            st.success(f"🩻 Máscara generada y guardada en 'datos/mask_new/'")

        image = Image.open(uploaded_image).convert("RGB")

        with st.spinner("🧠 Procesando con modelo TFLite..."):
            interpreter = load_model()
            image_batch, processed_image = preprocess_image(image)

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
        st.image(image, caption="🖼️ Imagen original", use_container_width=True)
        st.image(processed_image, caption="🔍 Imagen procesada", use_container_width=True)
        st.image(processed_image, caption="🔍 Imagen procesada", use_column_width=True)
        guardar_resultado(
            nombre_archivo=uploaded_image.name,
            resultado=predicted_label,
            probabilidades=prediction
        )
        st.info("💾 Resultado almacenado en resultados.csv")

with col2:
    st.header("📊 Resultados & Evaluación")
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
        st.image("training_accuracy.png", caption="Precisión por época", use_container_width=True)

    if os.path.exists("evaluation_result.png"):
        st.subheader("🧪 Evaluación del modelo")
        st.image("evaluation_result.png", caption="Comparación: Etiquetas reales vs predichas", use_container_width=True)
        st.image("training_accuracy.png", caption="Precisión por época", use_column_width=True)

    if os.path.exists("evaluation_result.png"):
        st.subheader("🧪 Evaluación del modelo")
        st.image("evaluation_result.png", caption="Comparación: Etiquetas reales vs predichas", use_column_width=True)

import os
import numpy as np
import cv2
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

IMAGE_SIZE = (150, 150)
DATA_DIR = "datos/procesados_split/test"
MODEL_PATH = "models/model_int8.tflite"
LABELS = ["COVID", "Normal", "Viral_Pneumonia"]
label_map = {name: idx for idx, name in enumerate(LABELS)}

@dask.delayed
def process_image(image_path, mask_path, label_idx):
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None

    img = cv2.resize(img, IMAGE_SIZE)

    if os.path.exists(mask_path):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, IMAGE_SIZE) / 255.0
        img = img * mask[..., np.newaxis]
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0)
        cl = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

    img = img.astype(np.float32) / 255.0
    return img, label_idx

def load_images_and_labels():
    tasks = []
    for label_name in LABELS:
        label_idx = label_map[label_name]
        img_dir = Path(DATA_DIR) / label_name / "images"
        mask_dir = Path(DATA_DIR) / label_name / "masks"
        
        for img_file in img_dir.glob("*.png"):
            mask_file = mask_dir / img_file.name
            tasks.append(process_image(img_file, mask_file, label_idx))
    
    with ProgressBar():
        results = dask.compute(*tasks)
        
    images, labels = zip(*[res for res in results if res[0] is not None])
    return np.array(images), np.array(labels)

def preprocess_for_tflite(img, input_details):
    if input_details['dtype'] == np.int8:
        scale, zero_point = input_details['quantization']
        img = img / scale + zero_point
        img = np.round(img).astype(np.int8)
    return img

def predict_batch(interpreter, X):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_index = input_details["index"]
    output_index = output_details["index"]

    predictions = []
    for img in X:
        img = preprocess_for_tflite(img, input_details)
        interpreter.set_tensor(input_index, np.expand_dims(img, axis=0))
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        if output_details['dtype'] == np.int8:
            scale, zero_point = output_details['quantization']
            output = scale * (output.astype(np.float32) - zero_point)
        predictions.append(np.argmax(output))
    return predictions

if __name__ == "__main__":
    print("📥 Cargando imágenes...")
    X, y_true = load_images_and_labels()

    print("🧠 Cargando modelo TFLite...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    print("🔍 Ejecutando predicción...")
    y_pred = predict_batch(interpreter, X)

    print("\n📊 Clasificación:")
    print(classification_report(y_true, y_pred, target_names=LABELS))

    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Precisión total: {acc*100:.2f}%")

    # Gráfico
    plt.figure()
    plt.hist([y_true, y_pred], label=["Reales", "Predichos"], bins=3, align='left')
    plt.xticks(range(3), LABELS)
    plt.title("Distribución de etiquetas")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("evaluation_result.png")
    plt.show()

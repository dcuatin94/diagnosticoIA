import os
import dotenv
import json
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm 

dotenv.load_dotenv()
SOURCE_ROOT_DIR = os.path.join(os.getenv('DIR_DATA_BASE'), "procesados")
DEST_ROOT_DIR = os.path.join(os.getenv('DIR_DATA_BASE'), "procesados_split")
CLASSES = json.loads(os.getenv('LABELS'))

# Ratio para el conjunto de prueba (ej. 0.2 para 20% de prueba, 80% de entrenamiento)
TEST_SPLIT_RATIO = 0.2

# Semilla para la reproducibilidad de la división
RANDOM_SEED = 42

def setup_directories(dest_base_dir, class_names):
    """Crea la estructura de directorios train/test/class/images y masks."""
    for split_type in ['train', 'test']:
        for class_name in class_names:
            os.makedirs(Path(dest_base_dir) / split_type / class_name / 'images', exist_ok=True)
            os.makedirs(Path(dest_base_dir) / split_type / class_name / 'masks', exist_ok=True)
    print(f"Estructura de directorios creada en: {dest_base_dir}")

def copy_files(file_paths, source_base_dir, dest_base_dir, split_type, class_name):
    """Copia archivos de imagen y máscara al directorio de destino."""
    print(f"Copiando {len(file_paths)} archivos a {split_type}/{class_name}...")
    for img_path in tqdm(file_paths, desc=f"Copiando {split_type}/{class_name}"):
        # Rutas de origen completas
        original_img_full_path = img_path
        original_mask_full_path = Path(str(img_path).replace(
            str(Path(source_base_dir) / class_name / 'images'),
            str(Path(source_base_dir) / class_name / 'masks')
        ))

        # Rutas de destino
        dest_img_path = Path(dest_base_dir) / split_type / class_name / 'images' / img_path.name
        dest_mask_path = Path(dest_base_dir) / split_type / class_name / 'masks' / img_path.name

        shutil.copy(original_img_full_path, dest_img_path)
        if original_mask_full_path.exists():
            shutil.copy(original_mask_full_path, dest_mask_path)
        else:
            print(f"Advertencia: Máscara no encontrada para {img_path.name}. Solo se copiará la imagen.")

def main():
    if not Path(SOURCE_ROOT_DIR).is_dir():
        print(f"Error: El directorio de origen '{SOURCE_ROOT_DIR}' no existe. Por favor, verifica la ruta.")
        return

    # Limpiar o crear el directorio de destino
    if Path(DEST_ROOT_DIR).exists():
        print(f"Eliminando el directorio de destino existente: {DEST_ROOT_DIR}")
        shutil.rmtree(DEST_ROOT_DIR)
    os.makedirs(DEST_ROOT_DIR, exist_ok=True)

    setup_directories(DEST_ROOT_DIR, CLASSES)

    print("\nIniciando la división y copia de datos...")
    for class_name in CLASSES:
        print(f"\nProcesando clase: {class_name}")
        class_images_dir = Path(SOURCE_ROOT_DIR) / class_name / "images"

        if not class_images_dir.is_dir():
            print(f"Advertencia: Directorio de imágenes para '{class_name}' no encontrado: {class_images_dir}. Saltando esta clase.")
            continue

        all_images_in_class = list(class_images_dir.glob("*.png"))
        if not all_images_in_class:
            print(f"Advertencia: No se encontraron imágenes PNG en {class_images_dir}. Saltando esta clase.")
            continue

        train_images, test_images = train_test_split(
            all_images_in_class, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED
        )

        print(f"Clase {class_name}: {len(train_images)} para entrenamiento, {len(test_images)} para prueba.")

        copy_files(train_images, SOURCE_ROOT_DIR, DEST_ROOT_DIR, 'train', class_name)
        copy_files(test_images, SOURCE_ROOT_DIR, DEST_ROOT_DIR, 'test', class_name)

    print("\n¡División de datos completada!")
    print(f"Los datos divididos están ahora en: {DEST_ROOT_DIR}")

if __name__ == "__main__":
    main()
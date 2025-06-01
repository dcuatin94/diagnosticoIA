import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm 


class SplitDataset():
    def __init__(self, source_root_dir, dest_root_dir, classes, test_split_ratio=0.2, random_seed=42):
        self.source_root_dir = source_root_dir
        self.dest_root_dir = dest_root_dir
        self.classes = classes
        self.test_split_ratio = test_split_ratio
        self.random_seed = random_seed

    def setup_directories(self, dest_base_dir, classes):
        """Crea la estructura de directorios train/test/class/images y masks."""
        classes = self.classes
        for split_type in ['train', 'test']:
            for class_name in classes:
                os.makedirs(Path(dest_base_dir) / split_type / class_name / 'images', exist_ok=True)
                os.makedirs(Path(dest_base_dir) / split_type / class_name / 'masks', exist_ok=True)
        print(f"Estructura de directorios creada en: {self.dest_root_dir}")

    def copy_files(self, file_paths, split_type, class_name):
        """Copia archivos de imagen y máscara al directorio de destino."""
        print(f"Copiando {len(file_paths)} archivos a {split_type}/{class_name}...")
        for img_path in tqdm(file_paths, desc=f"Copiando {split_type}/{class_name}"):
            # Rutas de origen completas
            original_img_full_path = img_path
            original_mask_full_path = Path(str(img_path).replace(
                str(Path(self.source_root_dir) / class_name / 'images'),
                str(Path(self.source_root_dir) / class_name / 'masks')
            ))

            # Rutas de destino
            dest_img_path = Path(self.dest_root_dir) / split_type / class_name / 'images' / img_path.name
            dest_mask_path = Path(self.dest_root_dir) / split_type / class_name / 'masks' / img_path.name

            # Copiar imagen
            shutil.copy(original_img_full_path, dest_img_path)

            # Copiar máscara si existe
            if original_mask_full_path.exists():
                shutil.copy(original_mask_full_path, dest_mask_path)
            else:
                print(f"⚠️  Advertencia: Máscara no encontrada para {img_path.name}")



    def ejecutar(self):
        if not Path(self.source_root_dir).is_dir():
            print(f"Error: El directorio de origen '{self.source_root_dir}' no existe. Por favor, verifica la ruta.")
            return

        # Limpiar o crear el directorio de destino
        if Path(self.dest_root_dir).exists():
            print(f"Eliminando el directorio de destino existente: {self.dest_root_dir}")
            shutil.rmtree(self.dest_root_dir)
        os.makedirs(self.dest_root_dir, exist_ok=True)

        self.setup_directories(dest_base_dir=self.dest_root_dir, classes=self.classes)

        print("\nIniciando la división y copia de datos...")
        for class_name in self.classes:
            print(f"\nProcesando clase: {class_name}")
            class_images_dir = Path(self.source_root_dir) / class_name / "images"

            if not class_images_dir.is_dir():
                print(f"Advertencia: Directorio de imágenes para '{class_name}' no encontrado: {class_images_dir}. Saltando esta clase.")
                continue

            # Cambia para incluir todos los archivos de imagen válidos
            all_images_in_class = list(class_images_dir.glob("*.png"))
            if not all_images_in_class:
                print(f"Advertencia: No se encontraron imágenes en {class_images_dir}. Saltando esta clase.")
                continue

            train_images, test_images = train_test_split(
                all_images_in_class, test_size=self.test_split_ratio, random_state=self.random_seed
            )

            print(f"Clase {class_name}: {len(train_images)} para entrenamiento, {len(test_images)} para prueba.")

            self.copy_files(train_images, 'train', class_name)
            self.copy_files(test_images, 'test', class_name)

        print("\n¡División de datos completada!")
        print(f"Los datos divididos están ahora en: {self.dest_root_dir}")

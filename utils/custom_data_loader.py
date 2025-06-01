import os
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.utils import Sequence, to_categorical # type: ignore
import random
import albumentations as A

class LungImageGenerator(Sequence):
    def __init__(self, base_dir, labels, image_size=(150, 150), batch_size=32, shuffle=True, augment=False):
        self.base_dir = Path(base_dir)
        self.labels = labels
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.data = self._load_paths()
        self.on_epoch_end()

        # Definir el pipeline de Albumentations al inicializar el generador
        self.augmentor = A.Compose([
            A.HorizontalFlip(p=0.5), # Volteo horizontal con 50% de probabilidad
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0), # Rotación +/- 15 grados
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=5,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ), # Traslación, escala y rotación combinadas
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3), # Brillo y contraste
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        ])


    def _load_paths(self):
        data = []
        for label_index, label_name in enumerate(self.labels):
            img_dir = self.base_dir / label_name / "images"
            mask_dir = self.base_dir / label_name / "masks"

            if not img_dir.is_dir():
                print(f"Advertencia: Directorio de imágenes no encontrado: {img_dir}")
                continue
            if not mask_dir.is_dir():
                print(f"Advertencia: Directorio de máscaras no encontrado: {mask_dir}")
                continue 
    
            for img_path in img_dir.glob("*.png"):
                mask_path = mask_dir / img_path.name
                if not mask_path.exists():
                    print(f"Advertencia: Máscara no encontrada para {img_path}. Se procesará sin máscara.")
                    mask_path = None # Indicar que no hay máscara para esta imagen
                data.append((str(img_path), str(mask_path) if mask_path else None, label_index))
        return data

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data) // self.batch_size

    def _augment(self, img):
        #Aplica las transformaciones de Albumentations a una imagen.
        
        augmented_image = self.augmentor(image=img)['image']
        return augmented_image

    def __getitem__(self, idx):
        batch = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []

        for img_path, mask_path, label in batch:
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Asegurarse de que la imagen tenga 3 canales (RGB)
            if len(img.shape) == 2: # Si es escala de grises
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4: # Si tiene canal alfa (RGBA)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            img = cv2.resize(img, self.image_size)

            # Manejo de máscaras
            if mask_path and os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    cl = clahe.apply(l)
                    img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
                else:
                    mask = cv2.resize(mask, self.image_size)
                    mask = mask / 255.0
                    img = img * mask[..., np.newaxis]
            else:
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
            
            # Normalizar y aplicar augmentación
            img = np.clip(img, 0, 255).astype(np.uint8)
            if self.augment:
                img = self._augment(img)

            img = img.astype(np.float32) / 255.0
            X.append(img)
            y.append(label)

        return np.array(X), to_categorical(y, num_classes=len(self.labels))
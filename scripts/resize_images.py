import os
import cv2
import json
import dotenv
import numpy as np
from pathlib import Path

class ResizeImages:
    """Clase para redimensionar imágenes y aplicar máscaras."""
    def __init__(self, input_path, output_path, target_size):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.target_size = target_size
        self.input_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def apply_mask(self, image, mask):
        """Aplica la máscara binaria sobre la imagen original, ajustando tamaño y tipo."""
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask_bin = mask_bin.astype(np.uint8)
        masked_image = cv2.bitwise_and(image, image, mask=mask_bin)
        return masked_image

    def resize_and_save(self, image_path, mask_path, output_path):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"⚠️ No se pudo leer la imagen: {image_path}")
            return

        if mask_path and mask_path.exists():
            mask = cv2.imread(str(mask_path))
            if mask is not None:
                image = self.apply_mask(image, mask)
            else:
                print(f"⚠️ No se pudo leer la máscara: {mask_path}")

        # Redimensionar y guardar
        resized = cv2.resize(image, self.target_size)
        cv2.imwrite(str(output_path), resized)
        print(f"✅ Guardada: {output_path}")

    def process_directory(self):
        for label in os.listdir(self.input_path):
            input_images_dir = self.input_path / label / "images"
            input_masks_dir = self.input_path / label / "masks"
            output_images_dir = self.output_path / label / "images"
            output_masks_dir = self.output_path / label / "masks"
            output_images_dir.mkdir(parents=True, exist_ok=True)
            output_masks_dir.mkdir(parents=True, exist_ok=True)

            # Procesar imágenes
            if input_images_dir.exists():
                for img_file in input_images_dir.glob("*.*"):
                    if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                        continue
                    mask_file = input_masks_dir / (img_file.stem + ".png")
                    output_file = output_images_dir / img_file.name
                    self.resize_and_save(img_file, mask_file, output_file)

            # Procesar máscaras si existen
            if input_masks_dir.exists():
                for mask_file in input_masks_dir.glob("*.png"):
                    output_mask_file = output_masks_dir / mask_file.name
                    mask = cv2.imread(str(mask_file))
                    if mask is not None:
                        resized_mask = cv2.resize(mask, self.target_size)
                        cv2.imwrite(str(output_mask_file), resized_mask)
                        print(f"✅ Máscara guardada: {output_mask_file}")
                    else:
                        print(f"⚠️ No se pudo leer la máscara: {mask_file}")
                        
    def ejecutar(self):
        """Ejecuta el proceso de redimensionamiento y guardado."""
        self.process_directory()
        print("🏁 Proceso finalizado.")


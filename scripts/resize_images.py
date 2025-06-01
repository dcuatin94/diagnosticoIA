import os
import cv2
import json
import dotenv
import numpy as np
from pathlib import Path

dotenv.load_dotenv()
PATH_ORIGINAL = os.path.join(os.getenv('DIR_DATA_BASE'), "original")
PATH_PROCESADOS = os.path.join(os.getenv('DIR_DATA_BASE'), "procesados")
# Rutas
input_path = Path(PATH_ORIGINAL)
output_path = Path(PATH_PROCESADOS)
target_size = json.loads(os.getenv('IMAGE_SIZE'))

def apply_mask(image, mask):
    """Aplica la máscara binaria sobre la imagen original, ajustando tamaño y tipo."""
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_bin = mask_bin.astype(np.uint8)
    masked_image = cv2.bitwise_and(image, image, mask=mask_bin)
    return masked_image

def resize_and_save(image_path, mask_path, output_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"⚠️ No se pudo leer la imagen: {image_path}")
        return

    if mask_path and mask_path.exists():
        mask = cv2.imread(str(mask_path))
        if mask is not None:
            image = apply_mask(image, mask)
        else:
            print(f"⚠️ No se pudo leer la máscara: {mask_path}")

    # Redimensionar y guardar
    resized = cv2.resize(image, target_size)
    cv2.imwrite(str(output_path), resized)
    print(f"✅ Guardada: {output_path}")

def process_directory():
    for label in os.listdir(input_path):
        input_images_dir = input_path / label / "images"
        input_masks_dir = input_path / label / "masks"
        output_images_dir = output_path / label / "images"
        output_masks_dir = output_path / label / "masks"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_masks_dir.mkdir(parents=True, exist_ok=True)

        # Procesar imágenes
        if input_images_dir.exists():
            for img_file in input_images_dir.glob("*.*"):
                if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                    continue
                mask_file = input_masks_dir / (img_file.stem + ".png")
                output_file = output_images_dir / img_file.name
                resize_and_save(img_file, mask_file, output_file)

        # Procesar máscaras si existen
        if input_masks_dir.exists():
            for mask_file in input_masks_dir.glob("*.png"):
                output_mask_file = output_masks_dir / mask_file.name
                mask = cv2.imread(str(mask_file))
                if mask is not None:
                    resized_mask = cv2.resize(mask, target_size)
                    cv2.imwrite(str(output_mask_file), resized_mask)
                    print(f"✅ Máscara guardada: {output_mask_file}")
                else:
                    print(f"⚠️ No se pudo leer la máscara: {mask_file}")

if __name__ == "__main__":
    process_directory()
    print("🏁 Proceso finalizado.")

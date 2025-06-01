import os
import cv2
import numpy as np
from pathlib import Path

# Rutas
input_path = Path("datos/original")
output_path = Path("datos/procesados")
target_size = (150, 150)

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
        output_dir = output_path / label
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_images_dir.exists():
            print(f"⚠️ Directorio no encontrado: {input_images_dir}")
            continue
        input_label_dir = input_path / label / "images"
        output_label_dir = output_path
        output_label_dir.mkdir(parents=True, exist_ok=True)

        for img_file in input_images_dir.glob("*.*"):
            if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue
            # Asumimos que las máscaras son .png
            mask_file = input_masks_dir / (img_file.stem + ".png")
            output_file = output_dir / img_file.name
            resize_and_save(img_file, mask_file, output_file)

if __name__ == "__main__":
    process_directory()
    print("🏁 Proceso finalizado.")

import os
import cv2
from pathlib import Path

input_path = Path("datos/original")
output_path = Path("datos/procesados")
target_size = (150, 150)

def resize_and_save(input_path, output_path):
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"⚠️ No se pudo leer: {input_path}")
        return
    resized = cv2.resize(image, target_size)
    cv2.imwrite(str(output_path), resized)

def process_directory():
    # No agregues 'images' aquí, ya que tu estructura es COVID/images
    for label in os.listdir(input_path):
        input_label_dir = input_path / label / "images"
        output_label_dir = output_path
        output_label_dir.mkdir(parents=True, exist_ok=True)

        for img_file in input_label_dir.glob("*.png"):
            output_file = output_label_dir / img_file.name
            resize_and_save(img_file, output_file)
        for img_file in input_label_dir.glob("*.jpg"):
            output_file = output_label_dir / img_file.name
            resize_and_save(img_file, output_file)

if __name__ == "__main__":
    process_directory()
    print("✅ Imágenes redimensionadas y guardadas en:", output_path)

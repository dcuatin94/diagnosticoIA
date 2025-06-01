import os
import shutil
from pathlib import Path

# Directorio raíz donde están las clases
root_dir = Path("datos/procesados")

# Recorre cada clase
for class_dir in root_dir.iterdir():
    if class_dir.is_dir():
        images_subdir = class_dir / "images"
        if images_subdir.exists():
            for img_file in images_subdir.iterdir():
                if img_file.is_file():
                    dest = class_dir / img_file.name
                    print(f"Moviendo {img_file} -> {dest}")
                    shutil.move(str(img_file), str(dest))
            # Elimina la subcarpeta vacía
            images_subdir.rmdir()
print("✅ Todas las imágenes han sido movidas a las carpetas de clase.")

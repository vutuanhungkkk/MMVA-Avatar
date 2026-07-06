from PIL import Image
from pathlib import Path

# Các định dạng ảnh hỗ trợ
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

current_dir = Path(".")

for image_path in current_dir.iterdir():
    if image_path.suffix.lower() in image_extensions:
        try:
            with Image.open(image_path) as img:
                resized = img.resize((320, 320), Image.Resampling.LANCZOS)
                resized.save(image_path)
                print(f"✓ Resized: {image_path.name}")
        except Exception as e:
            print(f"✗ Failed: {image_path.name} - {e}")

print("Done!")
import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageChops

# === Configuration ===
INPUT_ROOT  = "/Users/christianchen/VSCode_Python/Stat21/mnist_png/fine_tuning"
OUTPUT_ROOT = "/Users/christianchen/VSCode_Python/Stat21/mnist_png/augmented"
N_AUG       = 20  # number of augmented copies per image

# === Augmentation Functions ===
def augment_image(img: Image.Image) -> Image.Image:
    """
    Apply random augmentations: rotation, shift
    """
    # 1) Random rotation
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, fillcolor=0)

    # 2) Random shift
    max_shift = 4
    x_shift = random.randint(-max_shift, max_shift)
    y_shift = random.randint(-max_shift, max_shift)
    img = ImageChops.offset(img, x_shift, y_shift)

    

    return img


def process_folder(input_dir: str, output_dir: str, n_augs: int):
    """
    Generate n_augs augmented images for each image in input_dir,
    saving them into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        base, ext = os.path.splitext(fname.lower())
        if ext not in {".png", ".jpg", ".jpeg"}:
            continue

        img_path = os.path.join(input_dir, fname)
        try:
            img = Image.open(img_path).convert("L").resize((28, 28))
        except Exception as e:
            print(f"  ❌ Skipping {fname}: {e}")
            continue

        for i in range(n_augs):
            aug = augment_image(img)
            out_name = f"{base}_aug{i+1}{ext}"
            aug.save(os.path.join(output_dir, out_name))
        print(f"  ✅ {fname} → {n_augs} augmented images in '{output_dir}'")


if __name__ == "__main__":
    print(f"Augmenting images from '{INPUT_ROOT}' to '{OUTPUT_ROOT}', {N_AUG} per image")

    for label in sorted(os.listdir(INPUT_ROOT)):
        in_label_dir = os.path.join(INPUT_ROOT, label)
        if not os.path.isdir(in_label_dir):
            continue

        out_label_dir = os.path.join(OUTPUT_ROOT, label)
        print(f"\nProcessing label '{label}'...")
        process_folder(in_label_dir, out_label_dir, N_AUG)

    print("\nAll done!")

from PIL import Image
import os

orig_path = "images/original"
hr_path = "images/high_res"
lr_path = "images/low_res"

# Create folders if they don't exist
os.makedirs(hr_path, exist_ok=True)
os.makedirs(lr_path, exist_ok=True)

# Clear old files
def clear_folder(folder):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

clear_folder(hr_path)
clear_folder(lr_path)

# Choose a consistent crop size (MUST be divisible by 2)
crop_size = 432 # 432x432 high-res crop
scale = 2
low_size = crop_size // scale # 288×288 low-res crop

# Loop through original images and create HR/LR pairs
for filename in os.listdir(orig_path):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(orig_path, filename)
        img = Image.open(img_path).convert("RGB")

        # Center crop the image to a square
        w, h = img.size
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        img_cropped = img.crop((left, top, right, bottom))

        # Save high-res image
        img_cropped.save(os.path.join(hr_path, filename))

        # Downscale to low-res using bicubic interpolation
        lr_img = img_cropped.resize((low_size, low_size), Image.BICUBIC)
        lr_img.save(os.path.join(lr_path, filename))

print(f"Prepared {crop_size}×{crop_size} HR and {low_size}×{low_size} LR image pairs.")

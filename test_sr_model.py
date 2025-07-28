import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from train_sr_edsr import EDSR # reuse model class

LOW_RES_DIR = "images/low_res"
HIGH_RES_DIR = "images/high_res"
MODEL_PATH = "models/sr_model.pth"
UPSCALE = 2  # model was trained on 2×

# load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EDSR(scale_factor=UPSCALE).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


# load a test image
filename = os.listdir(LOW_RES_DIR)[10]

lr_image = Image.open(os.path.join(LOW_RES_DIR, filename)).convert("RGB")
hr_image = Image.open(os.path.join(HIGH_RES_DIR, filename)).convert("RGB") 

# Convert to tensor
to_tensor = transforms.ToTensor()
lr_tensor = to_tensor(lr_image).unsqueeze(0).to(device)  # Add batch dimension

# Run inference
with torch.no_grad():
    sr_tensor = model(lr_tensor).clamp(0, 1) # prevents overflow or underflow <- added after debugging

# Convert tensors back to PIL image
to_pil = transforms.ToPILImage()
sr_image = to_pil(sr_tensor.squeeze().cpu())

# Show all images 
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(lr_image)
axs[0].set_title("Low-Res Input")
axs[0].axis("off")

axs[1].imshow(sr_image)
axs[1].set_title("Model Output (SR)")
axs[1].axis("off")

axs[2].imshow(hr_image)
axs[2].set_title("High-Res Target")
axs[2].axis("off")

plt.tight_layout()
plt.show()

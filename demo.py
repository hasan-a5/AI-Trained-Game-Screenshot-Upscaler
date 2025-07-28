import gradio as gr
from PIL import Image
import torch
from torchvision import transforms
from train_sr_edsr import EDSR  # Make sure path and class names match
import os

# Set model paths and upscale factor
MODEL_PATH = "models/sr_model.pth"
UPSCALE = 2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = EDSR(scale_factor=UPSCALE).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Define image transform
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# Inference function
def super_resolve(image_path):
    # Open low-res image
    lr_img = Image.open(image_path).convert("RGB")
    lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)

    # Run model
    with torch.no_grad():
        sr_tensor = model(lr_tensor).clamp(0, 1)

    # Convert to PIL image
    sr_img = to_pil(sr_tensor.squeeze().cpu())
    return sr_img

# Gradio UI
title = "Game Screenshot Super Resolution"
description = "Upload a low-res game screenshot and get an enhanced version using a EDSR-based model."

demo = gr.Interface(
    fn=super_resolve,
    inputs=gr.Image(type="filepath", label="Upload Low-Res Screenshot", sources=["upload"]),
    outputs=gr.Image(type="pil", label="Super-Resolved Output"),
    title=title,
    description=description,
    allow_flagging="never"
)


if __name__ == "__main__":
    demo.launch()

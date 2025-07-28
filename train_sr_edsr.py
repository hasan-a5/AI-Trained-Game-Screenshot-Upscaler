import os
from PIL import Image # For loading images
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from pytorch_msssim import ssim # SSIM loss function

LOW_RES_DIR = "images/low_res"
HIGH_RES_DIR = "images/high_res"
MODEL_SAVE_PATH = "models/sr_model.pth"

# Hyperparameters
EPOCHS = 350 # Number of training cycles through the dataset
BATCH_SIZE = 4 # Number of images processed per batch
LEARNING_RATE = 1e-4 # How fast the model learns
UPSCALE = 2

# Data Augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # Randomly flip image horizontally
    transforms.RandomRotation(10), # Randomly rotate image ±10 degrees
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.01), # Slight color/brightness variation
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)), # Random zoom/shift
    transforms.ToTensor() # Convert PIL image to PyTorch tensor
])

# Dataset class to load paired LR/HR images
class SuperResDataset(torch.utils.data.Dataset):
    def __init__(self, lr_dir, hr_dir, transform):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        # Load image file names from low-res directory
        self.filenames = [f for f in os.listdir(lr_dir) if f.lower().endswith(("jpg", "png", "jpeg"))]

    def __len__(self): 
        return len(self.filenames) # Number of image pairs

    def __getitem__(self, idx): # tells PyTorch how to load one image pair
        fname = self.filenames[idx]
        # Load LR and HR images
        lr_img = Image.open(os.path.join(self.lr_dir, fname)).convert("RGB")
        hr_img = Image.open(os.path.join(self.hr_dir, fname)).convert("RGB")

        # Ensure the same random transform is applied to both images
        seed = torch.seed()
        torch.manual_seed(seed)
        lr_tensor = self.transform(lr_img)
        torch.manual_seed(seed)
        hr_tensor = self.transform(hr_img)
        return lr_tensor, hr_tensor


# EDSR model
class ResidualBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        # Each residual block has two convolution layers and a ReLU
        self.block = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1)
        )

    def forward(self, x):
        # Residual connection (input + output)
        return x + self.block(x)

class EDSR(nn.Module):
    def __init__(self, scale_factor=2, n_resblocks=16, n_feats=64):
        super().__init__()
        # Initial convolution (input to feature maps)
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)

        # Stack of residual blocks (deep part of the network)
        self.body = nn.Sequential(
            *[ResidualBlock(n_feats) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, 3, padding=1)
        )

        # Upsampling using PixelShuffle
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale_factor ** 2, 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(n_feats, 3, 3, padding=1)
        )

    def forward(self, x):
        x = self.head(x) # Initial conv
        res = self.body(x) # Pass through resblocks
        x = x + res # Global skip connection
        return self.upsample(x) # Upscale and return

    
# Perceptual Loss class using VGG16
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use the first few layers of pretrained VGG16 (feature extractor)
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:9].eval()
        for param in vgg.parameters():
            param.requires_grad = False # Freeze VGG weights
        self.vgg = vgg
        self.criterion = nn.MSELoss()
        # Normalization for VGG input
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, sr, hr):
        # Normalize each image in batch before passing through VGG
        sr_norm = torch.stack([self.normalize(x) for x in sr])
        hr_norm = torch.stack([self.normalize(x) for x in hr])
        # Compare VGG features
        return self.criterion(self.vgg(sr_norm), self.vgg(hr_norm))

# SSIM loss wrapper class
class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # SSIM returns similarity — we want to minimize (1 - similarity)
        return 1 - ssim(pred, target, data_range=1.0, size_average=True)
    
# Custom Gradient Loss class
class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss() # Use L1 loss (mean absolute error) to compare gradients

    def forward(self, pred, target):
        # Compute gradients for prediction and target
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

        # Compute total gradient loss (horizontal + vertical)
        loss = self.l1(pred_dx, target_dx) + self.l1(pred_dy, target_dy)
        return loss

# Train function
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # picks gpu if available
    print("Using device:", device)

    # Load dataset with augmentations applied
    dataset = SuperResDataset(LOW_RES_DIR, HIGH_RES_DIR, transform)

    # Batch and shuffle data; pin_memory speeds up data transfer to GPU
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    model = EDSR(scale_factor=UPSCALE).to(device) # Initialize model and move it to the GPU/CPU

    pixel_loss = nn.MSELoss()
    ssim_loss = SSIMLoss()
    perceptual_loss = PerceptualLoss().to(device)
    gradient_loss = GradientLoss().to(device)


    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam optimizer (adaptive learning rate)

    for epoch in range(EPOCHS): # sends each image batch to GPU, gets model predictions, computes loss, adjusts the model weights with backprop

        total_loss = 0.0

        # Process one batch at a time
        for lr_img, hr_img in loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            # Clamp outputs to [0, 1] range to match input expectations
            preds = model(lr_img).clamp(0, 1)

            # Combine multiple losses with carefully tuned weights
            loss = (1.0 * pixel_loss(preds, hr_img) + 0.3 * ssim_loss(preds, hr_img) + 0.001 * perceptual_loss(preds, hr_img) + 0.1 * gradient_loss(preds, hr_img))

            # Backpropagation step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}") # Prints total loss for that epoch

    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
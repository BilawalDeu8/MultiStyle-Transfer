#DenseNet Evaluation 

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as ssim
import math
import torch.nn.functional as F

# Load pre-trained DenseNet model and extract feature layers
class NeuralStyleTransfer(nn.Module):
    def __init__(self):
        super(NeuralStyleTransfer, self).__init__()
        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).features
        self.c_feature_layers = [8, 10]  # Content layers
        self.s_feature_layers = [4, 6]   # Style layers
  
    def forward(self, x):
        c_features = []
        s_features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if layer_num in self.c_feature_layers:
                c_features.append(x)
            elif layer_num in self.s_feature_layers:
                s_features.append(x)
        return c_features, s_features

# Function to load images and convert to tensor
def load_image(path, transform):
    img = Image.open(path)
    return transform(img).to('cuda')

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])

# Load two style images
style1 = load_image('path/to/file', transform)
style2 = load_image('path/to/file', transform)

content = load_image('path/to/file', transform)

# Using the content image as the start image
generated = load_image('path/to/file', transform)

# Displaying the images
fig = plt.figure(figsize=(10, 5))

plt.subplot(1,3,1)
plt.imshow(style1.permute(1,2,0).cpu()) # (h,w,c)
plt.title('Style Image 1')

plt.subplot(1,3,2)
plt.imshow(style2.permute(1,2,0).cpu()) # (h,w,c)
plt.title('Style Image 2')

plt.subplot(1,3,3)
plt.imshow(generated.permute(1,2,0).cpu())
plt.title('Generated Image')

plt.show()

# Set the generated image values to be tracked and modified while training
generated.requires_grad_(True)

# Set up the model
model = NeuralStyleTransfer()
model.to('cuda')
model.eval()

# Hyperparameters
step_count = 1000  # Increase steps for better results
learning_rate = 0.005  # Slightly increase learning rate
alpha = 1  # Content weight (keep as is)
beta = 0.1  # Increase style weight for better style application

# Custom optimizer for generated image
optimizer = torch.optim.Adam([generated], lr=learning_rate) 

# Standard Gram Matrix Loss function (no normalization)
def gram_matrix(features):
    bs, c, h, w = features.shape
    features = features.view(c, h * w)
    return torch.mm(features, features.T)

# PSNR function
def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

# SSIM function
'''
def calculate_ssim(img1, img2, win_size=3):
    img1 = img1.squeeze().cpu().numpy().transpose(1, 2, 0)
    img2 = img2.squeeze().cpu().numpy().transpose(1, 2, 0)

    # Ensure images are at least 7x7 for default win_size (if necessary)
    if img1.shape[0] < win_size or img1.shape[1] < win_size:
        print(f"Warning: Image size {img1.shape} is too small for the default win_size. Using smaller win_size.")
        win_size = 3  # Choose a smaller window size

    return ssim(img1, img2, multichannel=True, win_size=win_size)
'''

# Training loop
for i in tqdm(range(step_count)):
    # Get features from content, style1, style2, and generated images
    _, style1_features = model(style1.unsqueeze(0))
    _, style2_features = model(style2.unsqueeze(0))
    content_features, _ = model(content.unsqueeze(0))
    c_generated_features, s_generated_features = model(generated.unsqueeze(0))
  
    # Content loss: squared difference between content features
    content_loss = 0
    for cf, gf in zip(content_features, c_generated_features):
        content_loss += torch.sum((cf - gf)**2)

    # Style loss using standard Gram matrix loss for both style images
    style_loss = 0
    # First style image loss
    for sf, gf in zip(style1_features, s_generated_features):
        s_gram = gram_matrix(sf)
        g_gram = gram_matrix(gf)
        style_loss += torch.sum((s_gram - g_gram)**2)

    # Second style image loss
    for sf, gf in zip(style2_features, s_generated_features):
        s_gram = gram_matrix(sf)
        g_gram = gram_matrix(gf)
        style_loss += torch.sum((s_gram - g_gram)**2)

    # Combine the style losses (average of the two)
    style_loss = style_loss / 2

    # Total loss
    loss = alpha * content_loss + beta * style_loss

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print content and style loss individually
    if (i+1) % 50 == 0:  # Print every 50 steps
        print(f"Step [{i+1}/{step_count}], Content Loss: {content_loss.item()}, Style Loss: {style_loss.item()}")

# Display the generated image after training
plt.imshow(generated.permute(1,2,0).cpu().detach().numpy())
plt.title('Generated Image')
plt.show()

# Evaluate the metrics
content_img = content.cpu().detach()
generated_img = generated.cpu().detach()

# Calculate PSNR
psnr_value = psnr(content_img, generated_img)
print(f"PSNR: {psnr_value.item()} dB")



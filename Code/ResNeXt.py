#ResNeXt with performance metrics

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np

# Metrics-related imports
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Performance Metrics Class
class StyleTransferMetrics:
    @staticmethod
    def calculate_ssim(original, stylized):
        """Calculate Structural Similarity Index (SSIM)"""
        original_np = original.squeeze().permute(1,2,0).cpu().numpy()
        stylized_np = stylized.squeeze().permute(1,2,0).cpu().numpy()
        
        # Ensure images are in the range [0, 1]
        original_np = (original_np - original_np.min()) / (original_np.max() - original_np.min())
        stylized_np = (stylized_np - stylized_np.min()) / (stylized_np.max() - stylized_np.min())
        
        # Compute SSIM for each color channel
        ssim_scores = []
        for i in range(original_np.shape[2]):
            ssim_scores.append(ssim(original_np[:,:,i], stylized_np[:,:,i], 
                                    data_range=stylized_np[:,:,i].max() - stylized_np[:,:,i].min()))
        
        return np.mean(ssim_scores)
    
    @staticmethod
    def calculate_psnr(original, stylized):
        """Calculate Peak Signal-to-Noise Ratio (PSNR)"""
        original_np = original.squeeze().permute(1,2,0).cpu().numpy()
        stylized_np = stylized.squeeze().permute(1,2,0).cpu().numpy()
        
        # Ensure images are in the range [0, 255]
        original_np = (original_np * 255).astype(np.uint8)
        stylized_np = (stylized_np * 255).astype(np.uint8)
        
        return psnr(original_np, stylized_np, data_range=255)
    
    @staticmethod
    def calculate_content_preservation(content_features, generated_features):
        """Calculate content preservation by comparing feature maps"""
        import torch.nn.functional as F
        content_preservation = 0
        for cf, gf in zip(content_features, generated_features):
            # Normalized feature map comparison
            content_preservation += F.mse_loss(
                F.normalize(cf, p=2, dim=1), 
                F.normalize(gf, p=2, dim=1)
            ).item()
        
        return 1 / (1 + content_preservation)  # Inverse of loss for score-like metric

# Load pre-trained ResNeXt model and extract feature layers
class NeuralStyleTransfer(nn.Module):
    def __init__(self):
        super(NeuralStyleTransfer, self).__init__()
        # Use ResNeXt model (50 layers with 32x4d cardinality as example)
        self.model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        
        # ResNeXt has a 'conv1' (stem) followed by layers in 'layer1', 'layer2', 'layer3', etc.
        self.c_feature_layers = ['layer2', 'layer3']  # Content feature layers
        self.s_feature_layers = ['layer1']  # Style layers are usually from earlier layers
  
    def forward(self, x):
        c_features = []
        s_features = []
        
        # Forward through the stem layer (conv1)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        # Forward through layers and collect features
        for name, layer in self.model.named_children():
            if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                x = layer(x)
                # Collect content features from 'layer2' and 'layer3'
                if name in self.c_feature_layers:
                    c_features.append(x)
                # Collect style features from 'layer1'
                elif name in self.s_feature_layers:
                    s_features.append(x)
                
        return c_features, s_features

# Function to load images and convert to tensor
def load_image(path, transform):
    img = Image.open(path)
    return transform(img).to('cuda') 

# Existing code with metrics tracking
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

# Tracking best metrics
best_metrics = {
    'SSIM': 0,
    'PSNR': 0,
    'Content Preservation': 0
}

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

    # Periodically calculate and print metrics
    # Periodically calculate and print metrics
    if (i+1) % 100 == 0:
    # Calculate metrics
        current_metrics = {
            'SSIM': StyleTransferMetrics.calculate_ssim(content.detach(), generated.detach()),
            'PSNR': StyleTransferMetrics.calculate_psnr(content.detach(), generated.detach()),
            'Content Preservation': StyleTransferMetrics.calculate_content_preservation(
                content_features, c_generated_features
        )
    }
    
    # Update best metrics
        for key in best_metrics:
            if current_metrics[key] > best_metrics[key]:
                best_metrics[key] = current_metrics[key]
    
    # Print current step metrics
        print(f"Step [{i+1}/{step_count}], Content Loss: {content_loss.item()}, Style Loss: {style_loss.item()}")
        print("Current Metrics:")
        for key, value in current_metrics.items():
            print(f"{key}: {value:.4f}")

# Print final best metrics
print("\nBest Metrics Achieved:")
for key, value in best_metrics.items():
    print(f"{key}: {value:.4f}")

# Display the generated image after training
plt.figure(figsize=(10,5))
plt.imshow(generated.permute(1,2,0).cpu().detach())
plt.title('Generated Image')
plt.axis('off')
plt.show()

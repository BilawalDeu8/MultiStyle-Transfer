#VGG19 with Adaptive gram matrix loss evaluation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import requests
from io import BytesIO
import copy

# Function to load images from file paths or URLs
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

def load_image(image_path_or_url, imsize):
    if image_path_or_url.startswith('http'):  # If the input is a URL
        image = load_image_from_url(image_path_or_url)
    else:  # If the input is a file path
        image = Image.open(image_path_or_url).convert('RGB')
    image = transforms.Resize((imsize, imsize))(image)  # Ensure the image is the same size
    image = transforms.ToTensor()(image).unsqueeze(0)
    return image.to(device, torch.float)

# Paths or URLs for the images
style_image_paths_or_urls = [
    'path/to/file',
    'path/to/file'
]  
content_image_path_or_url = 'path/to/file'  # Path or URL to the content image

# device to run the network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# desired size of the output image
imsize = 512   # Use small size if no GPU

# Load and resize images
style_imgs = [load_image(img, imsize) for img in style_image_paths_or_urls]
content_img = load_image(content_image_path_or_url, imsize)

unloader = transforms.ToPILImage()  # Reconverts the tensor back into a PIL image

# Display utility
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # Clone the tensor to avoid changes on the original
    image = image.squeeze(0)      # Remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # Pause a bit so that plots are updated

# Display the style images and the content image
for i, style_img in enumerate(style_imgs):
    plt.figure()
    imshow(style_img, title=f'Style Image {i+1}')

plt.figure()
imshow(content_img, title='Content Image')

# Content loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

# Adaptive Gram Matrix Style Loss
class AdaptiveMSEStyleLoss(nn.Module):
    def __init__(self, target_features, device):
        super(AdaptiveMSEStyleLoss, self).__init__()
        self.targets = [gram_matrix(target).detach() for target in target_features]
        self.device = device
        
        # Calculate the "strength" of each style image's Gram matrix
        self.weights = self.calculate_weights(target_features)
    
    def calculate_weights(self, target_features):
        # Calculate the variance (strength) of each style image's Gram matrix
        weights = []
        for feature in target_features:
            G = gram_matrix(feature)
            strength = torch.var(G)  # Variance as a measure of strength
            weights.append(strength)
        
        # Normalize the weights to sum to 1 (optional)
        total_strength = sum(weights)
        normalized_weights = [weight / total_strength for weight in weights]
        return normalized_weights
    
    def forward(self, input):
        G = gram_matrix(input)
        loss = 0
        for idx, target in enumerate(self.targets):
            # Weighted loss for each style image
            weight = self.weights[idx]
            loss += weight * F.mse_loss(G, target)
        self.loss = loss
        return input

# Pretrained VGG network
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

# Normalization mean and standard deviation of RGB
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# Style and content layers
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# PSNR function definition
import torch
import torch.nn.functional as F

# Function to compute PSNR
def compute_psnr(img1, img2):
    # Ensure the images are in the range [0, 1]
    mse = F.mse_loss(img1, img2)
    psnr = 10 * torch.log10(1 / mse)
    return psnr

# Building the style transfer model
def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_imgs, content_img):
    normalization = Normalization(normalization_mean, normalization_std)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    i = 0  # Initialize counter for layers

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers_default:
            # Add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers_default:
            # Add Adaptive Gram matrix style loss:
            target_features = [model(style_img).detach() for style_img in style_imgs]
            style_loss = AdaptiveMSEStyleLoss(target_features, device)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], AdaptiveMSEStyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    # This line shows that the input image is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_imgs, input_img, num_steps=300,
                       style_weight=100000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_imgs, content_img)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            # Correct the values of the updated input image using 'clamp', not 'clamp_'
            with torch.no_grad():
                input_img.data = input_img.data.clamp(0, 1)
            
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            loss = style_score * style_weight + content_score * content_weight
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                
                # Calculate PSNR
                psnr_value = compute_psnr(content_img, input_img)
                print(f'PSNR: {psnr_value.item()} dB')

            return loss

        optimizer.step(closure)

    # A last correction without using in-place operation
    with torch.no_grad():
        input_img.data = input_img.data.clamp(0, 1)

    return input_img


# Execution of style transfer
input_img = content_img.clone()
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_imgs, input_img)

# Final PSNR calculation (after optimization completes)
final_psnr = compute_psnr(content_img, output)
print(f"Final PSNR: {final_psnr.item()} dB")

plt.figure()
imshow(output, title='Output Image')
plt.ioff()
plt.show()


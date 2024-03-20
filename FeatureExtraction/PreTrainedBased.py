"""
Flie containing function to extract feature using pre-trained models.

Creation date: 20/03/2024
Last modification: 20/03/2024
By: Mehdi 
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Define a function to load and preprocess an image
def preprocess_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

# Load pre-trained ResNet model
resnet = models.resnet50(pretrained=True)
# Remove the classification layer (fully connected layer) at the end
resnet = nn.Sequential(*list(resnet.children())[:-1])

def extract_features(image_path: str):
    """
    Function used to extract features from an image using ResNet. 

    :params image_path: str
        Path to the image for feature creation. 

    :return np.array
        Array containing 
    """
    image = preprocess_image(image_path, 224)  # Preprocess the image
    with torch.no_grad():
        features = resnet(image)  # Extract features using ResNet
    return features.squeeze().numpy()  # Remove batch dimension and convert to numpy array
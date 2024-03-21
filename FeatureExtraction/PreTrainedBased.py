"""
Flie containing function to extract feature using pre-trained models.

Creation date: 20/03/2024
Last modification: 20/03/2024
By: Mehdi 
"""
import os 
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def preprocess_image(image_path: str, image_size: int):
    """
    Function to load and preprocess one image.

    :params image_path: str or os.path object
        Path to the image
    :params image_size: int 
        Size of the image
    
    :return image wut batch dimension 
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension


def extract_features(image_path: str, model, img_size=128):
    """
    Function used to extract features from an image using ResNet. 

    :params image_path: str
        Path to the image for feature creation. 
    :params model: model
        Pre_trained model

    :return np.array
        Array containing features extracted
    """
    image = preprocess_image(image_path, img_size)  # Preprocess the image
    with torch.no_grad():
        features = model(image)  # Extract features using ResNet
    return features.squeeze().numpy()  # Remove batch dimension and convert to numpy array


def main(data_path, img_size=128, patients=None):
    """
    """    
    # Load pre-trained ResNet model
    resnet = models.resnet50(pretrained=True)
    # Remove the classification layer (fully connected layer) at the end
    resnet = nn.Sequential(*list(resnet.children())[:-1])

    if patients is None:
        patients = os.listdir(data_path)  # get list of patients 
        patients = [pat for pat in patients if '.csv' not in pat]
    pat_dic = {key: [] for key in patients} 

    for patient in patients:
        
        # get patient dir 
        patient_dir = os.path.join(data_path, patient)

        # get list of images corresponding to partient
        images = os.listdir(patient_dir)
        images = [img for img in images if ".jpg" in img]

        for img in images:
            img_path = os.path.join(patient_dir, img)
            # print("Image path: ", img_path)
            features = extract_features(image_path=img_path, 
                                        model=resnet, 
                                        img_size=img_size)
            
            # print("\nFeatures: ", features)
            pat_dic[patient].append(features)

    return pat_dic


if __name__ == "__main__":
    data_path = "./../data/trainset"
    dico = main(data_path, img_size=224)
    print(dico)
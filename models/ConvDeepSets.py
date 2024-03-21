"""
"""
import os
import pandas as pd 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class PatientDataset(Dataset):
    def __init__(self, root_dir, labels, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.patients = os.listdir(root_dir)
        self.patients = [pat for pat in self.patients if "csv" not in pat]
        self.labels = labels

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_folder = os.path.join(self.root_dir, self.patients[idx])
        images = []
        for filename in os.listdir(patient_folder):
            image_path = os.path.join(patient_folder, filename)
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        label = self.labels[self.patients[idx]]
        label_tensor = torch.tensor([label], dtype=torch.float32)
    
        return torch.stack(images), label_tensor
    

class SetEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(SetEncoder, self).__init__()
        """self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)"""
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Adjusted input size for 224x224 image
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        """print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)  # Adjusted input size for 224x224 image
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class SetFunction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SetFunction, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        # x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x
    

class ConvDeepSets(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_dim):
        super(ConvDeepSets, self).__init__()
        self.encoder = SetEncoder(input_channels, hidden_dim)
        self.set_function = SetFunction(hidden_dim, hidden_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.mean(x, dim=0, keepdim=True)  # Aggregating function
        x = self.set_function(x)
        return x


if __name__ == "__main__":
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    df_ann = pd.read_csv("./../data/clinical_annotation.csv")
    df_ann.set_index("ID", inplace=True)
    labels = df_ann.LABEL.to_dict()

    # Define the dataset
    dataset = PatientDataset(root_dir='./../data/trainset', 
                             transform=transform, 
                             labels=labels)

    # Define the DataLoader
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # define model 
    model = ConvDeepSets(3, 128, 1)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            print("Start")
            inputs, labels = batch
            inputs = torch.squeeze(inputs, dim=0)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute the loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Optimize
            optimizer.step()
            
            running_loss += loss.item()
            print("end")
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

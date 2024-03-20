"""
Convolutional AutoEncoder

Creation date: 14/03/2024
Last modification: 14/03/2024
By: Victor GIROU 
"""

import torch
import torch.nn as nn


class ConvAE(nn.Module):

    def __init__(self, n_channels, n_rows, n_cols, lr=3e-4):
        super(ConvAE, self).__init__()

        # data and network parameters 

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_channels = n_channels
        self.n_pixels = (self.n_rows)*(self.n_cols)

        self.lr = lr

        # encoder part

        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=4, stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=2, stride=(2, 2))

        # decoder part

        self.transconv1 = nn.ConvTranspose2d(3, 32, kernel_size=2,stride=(2, 2))
        self.transconv2 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=(2, 2), padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    
    def encoder(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return self.sigmoid(x)

    def decoder(self, z):

        z = self.transconv1(z)
        z = self.relu(z)
        z = self.transconv2(z)
        
        z = self.sigmoid(z)

        return z

    def forward(self, x):

        return self.decoder(self.encoder(x))
    
    def loss_function(self,x, y):

        mse_loss = self.criterion(y, x)
        return torch.mean(mse_loss)
    
    def train(self, data_train_loader, n_epochs):

        train_loss = []

        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch_idx, (data, _) in enumerate(data_train_loader):
                self.optimizer.zero_grad()
                
                y = self.forward(data)
                loss_ae = self.loss_function(data, y)
        
                loss_ae.backward()
                epoch_loss += loss_ae.item()
                self.optimizer.step()
                
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, epoch_loss / len(data_train_loader.dataset)))
            
            train_loss.append(epoch_loss / len(data_train_loader.dataset))
        
        return train_loss 
    
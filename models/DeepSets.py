"""
Main file containing models candidates for the classification task.

Creation date: 20/03/2024
Last modification: 20/03/2024
By: Mehdi 
"""
import torch
import torch.nn as nn
import pandas as pd 
import numpy as np

from torch.utils.data import DataLoader

class DeepSets(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialization of the deep set model. 

        :params input_dim: int
            Dimension of each element of the input. 
        :params hidden_dim: int 
            Dimension of hidden layers. 
        :parasm output_dim: int 
            Dimensionality of output
        """
        super(DeepSets, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 64),  
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU()
        )
        
        # Permutation-invariant layer
        self.invariant_layer = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16*3, output_dim), 
            nn.Sigmoid()
        )

    def forward(self, x):
        # Permute input data to ensure permutation invariance
        # x = x.permute(0, 2, 1)  # [batch_size, input_dim, num_elements]
        
        # Apply shared layers to each element of the set
        output = self.shared_layers(x)
        
        # Aggregate features across elements (sum or mean)
        mean = torch.mean(output, dim=1)  # Permutation-invariant aggregation
        std = torch.std(output, dim=1)
        max_, _ = torch.max(output, dim=1)
        
        y = torch.concat([mean, std, max_], dim=1)
        
        # Apply permutation-invariant layer
        y = self.invariant_layer(y)
        
        return y


def deepsets_loader(df):
    """
    
    """
    ids = pd.unique(df.ID)

    col = [str(i) for i in range(2048)]
    data = [(df[col].loc[df.ID == id].to_numpy(), df.loc[df.ID == id].LABEL.sum() > 0) for id in ids]
    data = [(torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)) for x, y in data]
    train_loader = DataLoader(data, shuffle=False)

    return train_loader
"""
Main file containing models candidates for the classification task.

Creation date: 20/03/2024
Last modification: 20/03/2024
By: Mehdi 
"""
import torch
import torch.nn as nn

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
            nn.Linear(input_dim, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Permutation-invariant layer
        self.invariant_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Permute input data to ensure permutation invariance
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, num_elements]
        
        # Apply shared layers to each element of the set
        x = self.shared_layers(x)
        
        # Aggregate features across elements (sum or mean)
        x = torch.mean(x, dim=2)  # Permutation-invariant aggregation
        
        # Apply permutation-invariant layer
        x = self.invariant_layer(x)
        
        return x
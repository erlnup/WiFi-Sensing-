import torch
import torch.nn as nn


class SignatureModel(nn.Module):
    def __init__(self, input_size, signature_dimensionality=10):
        super().__init__()
        self.linear = nn.Linear(input_size, signature_dimensionality)
    
    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

        


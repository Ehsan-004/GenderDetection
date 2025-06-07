import torch
import torch.nn as nn


class GenderModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.bloc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
            nn.LazyLinear(2)
        )
    
    def forward(self, x):
        return self.bloc(x)
        
        

from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class mlp_model(nn.Module):
    def __init__(self, in_ch: int, num_cls: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_ch, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, num_cls)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
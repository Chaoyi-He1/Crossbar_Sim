from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class mlp_model(nn.Module):
    def __init__(self, in_ch: int, num_cls: int):
        super().__init__()

        self.fc1 = nn.Linear(in_ch, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_cls)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, num_cls)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x
    
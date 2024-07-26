from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))
    
class ConvConvReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)))

class CNN_BN(nn.Module):
    def __init__(self, in_ch: int, num_cls: int, h: int, w: int):
        super().__init__()
        self.h = h
        self.w = w
        
        self.conv1 = ConvBNReLU(in_ch, 4)
        self.maxpool1 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv2 = ConvBNReLU(4, 8)
        self.maxpool2 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv3 = ConvBNReLU(8, 16)
        self.maxpool3 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv4 = ConvBNReLU(16, 32)
        self.maxpool4 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(32 * self.h * self.w, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_cls)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.maxpool3(x)
        
        x = self.conv4(x)
        x = self.maxpool4(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
class CNN_conv(nn.Module):
    def __init__(self, in_ch: int, num_cls: int, h: int, w: int):
        super().__init__()
        self.h = h
        self.w = w
        
        self.conv1 = ConvConvReLU(in_ch, 4)
        self.maxpool1 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv2 = ConvConvReLU(4, 8)
        self.maxpool2 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv3 = ConvConvReLU(8, 16)
        self.maxpool3 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv4 = ConvConvReLU(16, 32)
        self.maxpool4 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(32 * self.h * self.w, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_cls)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.maxpool3(x)
        
        x = self.conv4(x)
        x = self.maxpool4(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

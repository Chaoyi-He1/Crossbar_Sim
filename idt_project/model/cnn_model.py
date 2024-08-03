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
        self.conv_res = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x))) + self.conv_res(x)
    
class ConvConvReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 1)
        self.relu = nn.ReLU()
        self.conv_res = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x))) + self.conv_res(x)

class CNN_BN(nn.Module):
    def __init__(self, in_ch: int, num_cls: int, h: int, w: int):
        super().__init__()
        self.h = h
        self.w = w
        
        self.conv1 = ConvBNReLU(in_ch, 16)
        self.maxpool1 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv2 = ConvBNReLU(16, 32)
        self.maxpool2 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv3 = ConvBNReLU(32, 64)
        self.maxpool3 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv4 = ConvBNReLU(64, 128)
        self.maxpool4 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv5 = ConvBNReLU(128, 128)
        self.maxpool5 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(128 * self.h * self.w, 128)
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
        
        x = self.conv5(x)
        x = self.maxpool5(x)
        
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
        
        self.conv1 = ConvConvReLU(in_ch, 16)
        self.maxpool1 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv2 = ConvConvReLU(16, 32)
        self.maxpool2 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv3 = ConvConvReLU(32, 64)
        self.maxpool3 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv4 = ConvConvReLU(64, 128)
        self.maxpool4 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv5 = ConvConvReLU(128, 256)
        # self.maxpool5 = nn.MaxPool2d(2)
        # self.h, self.w = self.h // 2, self.w // 2
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(256 * self.h * self.w, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_cls)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.maxpool3(x)
        
        x = self.conv4(x)
        x = self.maxpool4(x)
        
        x = self.conv5(x)
        # x = self.maxpool5(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        
        return x

class CNN_conv_bone(nn.Module):
    def __init__(self, in_ch: int, num_cls: int, h: int, w: int):
        super().__init__()
        self.h = h
        self.w = w
        
        self.conv1 = ConvConvReLU(in_ch, 16, 5)
        self.maxpool1 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv2 = ConvConvReLU(16, 32)
        self.maxpool2 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv3 = ConvConvReLU(32, 64)
        self.maxpool3 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv4 = ConvConvReLU(64, 128)
        self.maxpool4 = nn.MaxPool2d(2)
        self.h, self.w = self.h // 2, self.w // 2
        
        self.conv5 = ConvConvReLU(128, 256)
        # self.maxpool5 = nn.MaxPool2d(2)
        # self.h, self.w = self.h // 2, self.w // 2
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(256 * self.h * self.w, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.maxpool3(x)
        
        x = self.conv4(x)
        x = self.maxpool4(x)
        
        x = self.conv5(x)
        x = self.maxpool5(x)
        
        x = self.flatten(x)
        
        x = self.fc2(self.relu(self.fc1(x)))

        return x

if __name__ == "__main__":
    model = CNN_BN(1, 30, 102, 62)
    print(model)
    model = CNN_conv(1, 30, 102, 62)
    print(model)
    
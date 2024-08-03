from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "sigmoid":
        return F.sigmoid
    if activation == "leaky_relu":
        return F.leaky_relu
    if activation == "tanh":
        return F.tanh
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def calculate_conv2d_padding(stride, kernel_size, d_in, d_out, dilation=1):
    """
    Calculate the padding value for a 2D convolutional layer.
    
    Arguments:
    - stride (int or tuple): The stride value(s) for the convolution.
    - kernel_size (int or tuple): The size of the convolutional kernel.
    - d_in (tuple): The input dimensions (height, width) of the feature map.
    - d_out (tuple): The output dimensions (height, width) of the feature map.
    - dilation (int or tuple): The dilation value(s) for the convolution. Default is 1.
    
    Returns:
    - padding (tuple): The padding value(s) (padding_h, padding_w) for the convolution.
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    h_in, w_in = d_in
    h_out, w_out = d_out
    h_k, w_k = kernel_size
    h_s, w_s = stride
    h_d, w_d = dilation

    padding_h = math.ceil(((h_out - 1) * h_s + h_k - h_in + (h_k - 1) * (h_d - 1)) / 2)
    padding_w = math.ceil(((w_out - 1) * w_s + w_k - w_in + (w_k - 1) * (w_d - 1)) / 2)
    assert padding_h >= 0 and padding_w >= 0, "Padding value(s) cannot be negative."

    padding = (padding_h, padding_w)
    return padding


class Conv2d_BN_Relu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(Conv2d_BN_Relu, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return super(Conv2d_BN_Relu, self).forward(x)


class ResBlock_2d(nn.Module):
    def __init__(self, in_channel: int, kernel_size: int = 3, stride: int = 1, 
                 in_dim: Tuple[int, int] = (256, 1024), dilation: int = 1, 
                 drop_path_ratio: float = 0.4) -> None:
        super(ResBlock_2d, self).__init__()
        pad = calculate_conv2d_padding(stride, kernel_size, in_dim, in_dim, dilation)
        self.conv1 = Conv2d_BN_Relu(in_channel, in_channel // 2, kernel_size=1)
        self.conv2 = Conv2d_BN_Relu(in_channel // 2, in_channel, kernel_size=kernel_size, 
                                    padding=pad, stride=stride, dilation=dilation)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.drop_path(self.conv2(self.conv1(x)))


class Conv2d_AutoEncoder(nn.Module):
    def __init__(self, in_dim: Tuple[int, int] = (256, 1024), 
                 in_channel: int = 1, drop_path: float = 0.4) -> None:
        super(Conv2d_AutoEncoder, self).__init__()
        self.in_channel = in_channel
        self.temp_dim = in_dim
        self.channel = 16
        pad = calculate_conv2d_padding(1, 5, self.temp_dim, self.temp_dim)
        self.conv1 = Conv2d_BN_Relu(self.in_channel, self.channel, kernel_size=5, padding=pad)
        pad = calculate_conv2d_padding(1, 5, self.temp_dim, tuple(element for element in self.temp_dim))
        self.conv2 = Conv2d_BN_Relu(self.channel, self.channel * 2, kernel_size=5, stride=1, padding=pad)
        self.channel *= 2
        self.temp_dim = tuple(element for element in self.temp_dim)

        self.ResNet = nn.ModuleList()
        res_params = list(zip([4, 4, 6, 6, 4], [7, 7, 5, 3, 3],   # num_blocks, kernel_size
                              [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]))   # stride, dilation
        # final channels = 512; final temp_dim = in_dim // (2^5) = in_dim // 32
        for i, (num_blocks, kernel_size, stride, dilation) in enumerate(res_params):
            self.ResNet.extend([ResBlock_2d(self.channel, kernel_size, stride, self.temp_dim, dilation,
                                            drop_path)
                                for _ in range(num_blocks)])
            if i != len(res_params) - 1:
                pad = calculate_conv2d_padding(2, kernel_size, self.temp_dim, 
                                               tuple(element // 2 for element in self.temp_dim), dilation)
                self.ResNet.append(Conv2d_BN_Relu(self.channel, self.channel * 2,
                                                  kernel_size, 2, pad, dilation))
                self.channel *= 2
                self.temp_dim = tuple(element // 2 for element in self.temp_dim)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: [L, 2, in_dim], 
        # L is the sequence_length in the following Transformer based AutoRegressive model
        # output: [L, Embedding], Embedding = 512

        # assert inputs.shape[1] == 2 and len(inputs.shape) == 3, "Input shape should be [B, 2, Embedding]"

        x = self.conv1(inputs)
        x = self.conv2(x)
        for block in self.ResNet:
            x = block(x)
        x = self.avgpool(x)
        return x.squeeze(-1).squeeze(-1)

class AutoEncoder_cls(nn.Module):
    def __init__(self, in_dim: Tuple[int, int] = (256, 1024), num_cls: int = 30,
                 in_channel: int = 1, drop_path: float = 0.4) -> None:
        super(AutoEncoder_cls, self).__init__()
        self.backbone = Conv2d_AutoEncoder(in_dim, in_channel, drop_path)
        self.fc = nn.Linear(512, num_cls)
    
    def forward(self, inputs: Tensor) -> Tensor:
        x = self.backbone(inputs)
        return self.fc(x)
import torch
from torch import nn
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import misc.util as util
import torch.nn.functional as F


def criterion(pred, target):
    loss = F.cross_entropy(pred, target)
    acc = torch.mean((torch.argmax(pred, dim=1) == target).float())
    return loss, acc

def quantize_regularize(model, alpha=0.1):
    central_params = torch.arange(0.0, 1.1, 0.1)
    reg_params = {}
    for c in central_params:
        if c == 0.0:
            reg_params[c] = [p.abs() < 0.05 for n, p in model.named_parameters() if "bn" not in n]
        if c == 1.0:
            reg_params[c] = [p.abs() > 0.95 for n, p in model.named_parameters() if "bn" not in n]
        else:
            reg_params[c] = [(c - 0.05 <= p.abs()) & (p.abs() < c + 0.05) for n, p in model.named_parameters() if "bn" not in n]
            
    

def train_one_epoch(model, optimizer, 
                    data_loader, device, epoch, print_freq=10, scaler=None):
    model.train()
    
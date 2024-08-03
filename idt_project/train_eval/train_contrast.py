import torch
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import misc.util as util
import torch.nn.functional as F


def criterion(pred):
    """
    pred: (B, dim), where B contains B / 2 pairs of samples
    Calculate the contrastive loss to maximize the similarity between the positive pairs
    and minimize the similarity between the negative pairs
    """
    B = pred.shape[0]
    # Calculate the similarity matrix
    sim_matrix = torch.matmul(pred, pred.t())
    # do softmax for each row, make the diagonal
    sim_matrix = sim_matrix - torch.eye(B).to(pred.device) * 1e6
    sim_matrix = F.log_softmax(sim_matrix, dim=-1)
    # Calculate the similarity between the positive pairs
    pos_sim = torch.diag(sim_matrix, diagonal=1)[::2]
    return -pos_sim.mean()

def quantize_regularize(model, device, alpha=0.5):
    central_params = torch.arange(0.0, 1.1, 0.1).to(device)
    reg_params = {}
    for c in central_params:
        if c == 0.0:
            reg_params[c] = [p.abs() < 0.05 for n, p in model.named_parameters()]
        if c == 1.0:
            reg_params[c] = [p.abs() > 0.95 for n, p in model.named_parameters()]
        else:
            reg_params[c] = [(c - 0.05 <= p.abs()) & (p.abs() < c + 0.05) for n, p in model.named_parameters()]
    
    reg_loss = 0.0
    for c, params in reg_params.items():
        reg_loss += alpha * torch.sum(torch.stack([torch.sum((p[params[i]].abs() - c).abs()) for i, (n, p) in enumerate(model.named_parameters()) if "bn" not in n]))
        
    return reg_loss  

def train_one_epoch(model, optimizer, alpha,
                    data_loader, device, epoch, print_freq=10, scaler=None, num_classes=30):
    model.train()
    header = 'Epoch: [{}]'.format(epoch)
    metric_logger = util.MetricLogger(delimiter="  ")
    
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device)
        # check if targets contains batch_size / 2 unique labels, targets is a tensor of shape (batch_size,)
        assert len(torch.unique(targets)) == targets.shape[0] // 2
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            output = model(images)
            loss = criterion(output)
            if epoch >= 80:
                loss += quantize_regularize(model, device, alpha)
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if epoch >= 80:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        metric_logger.update(lr=optimizer.param_groups[0]["lr"], loss=loss.item())

    metric_logger.synchronize_between_processes()
    return metric_logger.meters["loss"].global_avg
    
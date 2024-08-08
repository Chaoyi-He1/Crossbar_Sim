import torch
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
    
    all_preds, all_targets = [], []
    for images, target in metric_logger.log_every(data_loader, print_freq, header):
        images, target = images.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            output = model(images)
            loss, acc = criterion(output, target)
            if epoch >= 10:
                loss += quantize_regularize(model, device, alpha)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        metric_logger.update(lr=optimizer.param_groups[0]["lr"], loss=loss.item(), acc=acc.item())
        all_preds.append(torch.argmax(output, dim=-1).detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())
    
    all_preds, all_targets = np.concatenate(all_preds), np.concatenate(all_targets)
    cm = confusion_matrix(all_targets, all_preds)
    df_cm = pd.DataFrame(cm, index = [str(i) for i in range(num_classes)],
                         columns=[str(i) for i in range(num_classes)])
    plt.figure(figsize = (14, 10))
    fig = sn.heatmap(df_cm, annot=True).get_figure()
    plt.close()
    metric_logger.synchronize_between_processes()
    return metric_logger.meters["loss"].global_avg, metric_logger.meters["acc"].global_avg, fig
    
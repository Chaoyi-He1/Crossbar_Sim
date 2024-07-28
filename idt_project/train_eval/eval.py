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

def evaluate(model, data_loader, device, epoch, print_freq=10, scaler=None, num_classes=30):
    model.eval()
    header = "Test:"
    metric_logger = util.MetricLogger(delimiter="  ")
    
    all_preds, all_targets = [], []
    for images, target in metric_logger.log_every(data_loader, print_freq, header):
        images, target = images.to(device), target.to(device)
        
        with torch.no_grad():
            output = model(images)
            loss, acc = criterion(output, target)
        
        metric_logger.update(loss=loss.item(), acc=acc.item())
        all_preds.append(torch.argmax(output, dim=-1).detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    cm = confusion_matrix(all_targets, all_preds)
    df_cm = pd.DataFrame(cm, index = [str(i) for i in range(num_classes)],
                         columns=[str(i) for i in range(num_classes)])
    plt.figure(figsize = (14, 10))
    fig = sn.heatmap(df_cm, annot=True).get_figure()
    plt.close()
    
    metric_logger.synchronize_between_processes()
    return metric_logger["loss"].global_avg, metric_logger["acc"].global_avg, fig
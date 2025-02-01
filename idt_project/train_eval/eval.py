import torch
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import misc.util as util
import torch.nn.functional as F
from sklearn.manifold import TSNE

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
    return metric_logger.meters["loss"].global_avg, metric_logger.meters["acc"].global_avg, fig


def feature_extractor(model, data_loader, device, epoch, print_freq=10, scaler=None, num_classes=30):
    model.eval()
    header = "Test:"
    metric_logger = util.MetricLogger(delimiter="  ")

    all_preds, all_labels = [], []
    for images, target in metric_logger.log_every(data_loader, print_freq, header):
        images, target = images.to(device), target.to(device)
        
        with torch.no_grad():
            output = model(images)
        all_preds.append(output.detach().cpu().numpy())
        all_labels.append(target.detach().cpu().numpy())
        
    all_preds = np.vstack(all_preds)
    all_preds = (all_preds - all_preds.min()) / (all_preds.max() - all_preds.min()) * 255
    all_preds = all_preds.astype(np.uint8)
    all_labels = np.concatenate(all_labels).reshape(-1)
    # do t-SNE on the output

    tsne = TSNE(n_components=2, random_state=0)
    all_preds_ = tsne.fit_transform(all_preds)
    
    # generate the t-SNE plot
    colormap = plt.cm.get_cmap('tab10', 30)
    colors = [colormap(i) for i in all_labels]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(all_preds_[:, 0], all_preds_[:, 1], c=colors)
    return fig, all_preds_, all_labels, colors, colormap
    

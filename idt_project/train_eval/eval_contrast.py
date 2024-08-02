import torch
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import misc.util as util
import torch.nn.functional as F
from sklearn.manifold import TSNE


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
    sim_matrix.scatter_(1, torch.arange(B).to(pred.device).view(-1, 1), -1e12)
    sim_matrix = F.log_softmax(sim_matrix, dim=-1)
    # Calculate the similarity between the positive pairs
    pos_sim = torch.diag(sim_matrix, diagonal=1)[::2]
    return -pos_sim.mean()

def evaluate(model, data_loader, device, epoch, print_freq=10, scaler=None, num_classes=30):
    model.eval()
    header = "Test:"
    metric_logger = util.MetricLogger(delimiter="  ")
    
    all_preds, all_labels = [], []
    for images, target in metric_logger.log_every(data_loader, print_freq, header):
        images, target = images.to(device), target.to(device)
        
        with torch.no_grad():
            output = model(images)
            loss = criterion(output)
        
        metric_logger.update(loss=loss.item())
        all_preds.append(output.detach().cpu().numpy())
        all_labels.append(target.detach().cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels).reshape(-1)

    # do t-SNE on the output
    if epoch % 50 == 0:
        tsne = TSNE(n_components=2, random_state=0)
        all_preds = tsne.fit_transform(all_preds)
        
        # generate the t-SNE plot
        colormap = plt.cm.get_cmap('tab10', 30)
        colors = [colormap(i) for i in all_labels]
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.scatter(all_preds[:, 0], all_preds[:, 1], c=colors)
        plt.close(fig=fig)
        
        metric_logger.synchronize_between_processes()
        return metric_logger.meters["loss"].global_avg, fig
    
    metric_logger.synchronize_between_processes()
    return metric_logger.meters["loss"].global_avg
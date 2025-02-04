import torch
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import misc.util as util
import torch.nn.functional as F
from sklearn.manifold import TSNE


def criterion(pred, target, anomaly):
    # loss = F.cross_entropy(pred, target)
    # acc = torch.mean((torch.argmax(pred, dim=1) == target).float())
    device_pred, anomaly_pred = torch.split(pred, [30, 1], dim=1)
    loss = F.cross_entropy(device_pred, target, ignore_index=30)
    loss = loss + F.binary_cross_entropy_with_logits(anomaly_pred, anomaly.float().unsqueeze(1))
    acc = torch.sum((torch.argmax(device_pred, dim=1) == target).float() * (anomaly == 0).float()) / torch.sum((anomaly == 0).float())
    anomaly_acc = torch.mean(((torch.sigmoid(anomaly_pred) > 0.5).float() == anomaly.float().unsqueeze(1)).float())
    return loss, acc, anomaly_acc

def evaluate(model, data_loader, device, epoch, print_freq=10, scaler=None, num_classes=30):
    model.eval()
    header = "Test:"
    metric_logger = util.MetricLogger(delimiter="  ")
    
    all_preds, all_targets = [], []
    all_anomalies_preds, all_anomalies_targets = [], []
    for images, target, anomaly in metric_logger.log_every(data_loader, print_freq, header):
        images, target = images.to(device), target.to(device)
        anomaly = anomaly.to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                output = model(images)
                loss, acc, anomaly_acc = criterion(output, target, anomaly)
        
        metric_logger.update(loss=loss.item(), acc=acc.item(), anomaly_acc=anomaly_acc.item())
        all_preds.append(torch.argmax(output, dim=-1).detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())
        all_anomalies_preds.append((torch.sigmoid(output[:, -1]) > 0.5).detach().cpu().numpy())
        all_anomalies_targets.append(anomaly.detach().cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_anomalies_preds, all_anomalies_targets = np.concatenate(all_anomalies_preds), np.concatenate(all_anomalies_targets)
    
    cm1 = confusion_matrix(all_targets, all_preds)
    cm2 = confusion_matrix(all_anomalies_targets, all_anomalies_preds)
    df_cm1 = pd.DataFrame(cm1, index=[str(i) for i in range(num_classes)],
                            columns=[str(i) for i in range(num_classes)])
    df_cm2 = pd.DataFrame(cm2, index=["Normal", "Anomaly"],
                            columns=["Normal", "Anomaly"])
    # plot confusion matrix 1 and 2 in one figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 10))
    fig1 = sn.heatmap(df_cm1, annot=True, ax=axs[0]).get_figure()
    fig2 = sn.heatmap(df_cm2, annot=True, ax=axs[1]).get_figure()
    plt.close(fig1)
    plt.close(fig2)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return metric_logger.meters["loss"].global_avg, metric_logger.meters["acc"].global_avg, [fig1, fig2]


def feature_extractor(model, data_loader, device, epoch, print_freq=10, scaler=None, num_classes=30):
    model.eval()
    header = "Test:"
    metric_logger = util.MetricLogger(delimiter="  ")

    all_preds, all_labels = [], []
    for images, target, anomaly in metric_logger.log_every(data_loader, print_freq, header):
        images, target = images.to(device), target.to(device)
        anomaly = anomaly.to(device)
        
        with torch.no_grad():
            output = model(images)
        all_preds.append(output.detach().cpu().numpy())
        all_labels.append(target.detach().cpu().numpy())
        
    all_preds = np.vstack(all_preds)
    all_preds = (all_preds - all_preds.min()) / (all_preds.max() - all_preds.min()) * 230 + 10
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
    return fig, all_preds, all_labels, colors, colormap
    

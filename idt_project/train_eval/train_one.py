import torch
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import misc.util as util
import torch.nn.functional as F


def criterion(pred, target, anomaly):
    # loss = F.cross_entropy(pred, target)
    # acc = torch.mean((torch.argmax(pred, dim=1) == target).float())
    device_pred, anomaly_pred = torch.split(pred, [30, 1], dim=1)
    loss = F.cross_entropy(device_pred, target, ignore_index=30)
    loss = loss + F.binary_cross_entropy_with_logits(anomaly_pred, anomaly.float().unsqueeze(1))
    acc = torch.sum((torch.argmax(device_pred, dim=1) == target).float() * (anomaly == 0).float()) / torch.sum((anomaly == 0).float())
    anomaly_acc = torch.mean(((torch.sigmoid(anomaly_pred) > 0.5).float() == anomaly.float().unsqueeze(1)).float())
    return loss, acc, anomaly_acc

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
    all_anomalies_preds, all_anomalies_targets = [], []
    for images, target, anomaly in metric_logger.log_every(data_loader, print_freq, header):
        images, target = images.to(device), target.to(device)
        anomaly = anomaly.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            output = model(images)
            loss, acc, anomaly_acc = criterion(output, target, anomaly)
            if epoch >= 10 and alpha > 0:
                loss += quantize_regularize(model, device, alpha)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        metric_logger.update(lr=optimizer.param_groups[0]["lr"], loss=loss.item(), acc=acc.item(), anomaly_acc=anomaly_acc.item())
        all_preds.append(torch.argmax(output[:, :-1], dim=-1).detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())
        all_anomalies_preds.append((torch.sigmoid(output[:, -1]) > 0.5).detach().cpu().numpy())
        all_anomalies_targets.append(anomaly.detach().cpu().numpy())
    
    all_preds, all_targets = np.concatenate(all_preds), np.concatenate(all_targets)
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
    
    # cm = confusion_matrix(all_targets, all_preds)
    # df_cm = pd.DataFrame(cm, index = [str(i) for i in range(num_classes)],
    #                      columns=[str(i) for i in range(num_classes)])
    # plt.figure(figsize = (14, 10))
    # fig = sn.heatmap(df_cm, annot=True).get_figure()
    # plt.close()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return metric_logger.meters["loss"].global_avg, metric_logger.meters["acc"].global_avg, [fig1, fig2]
    
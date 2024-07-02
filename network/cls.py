import torch
from torch import nn
from torch import Tensor
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from typing import Iterable, List
import math
import os
import scipy.io as sio
import network.misc as misc
import regex as re
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import time, datetime

class Cls_dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder: str, with_CNN: bool = False):
        self.label_dict = {
            "pam2": 0,
            "pam4": 1,
            "psk4": 2,
            "psk8": 3,
            "qam4": 4,
            "qam16": 5,
            "qam64": 6,
        }
        
        self.with_CNN = with_CNN
        # extract all files in the folder with format "5core_xb_pn_xxxx_run.npy", where xxxx is the modulation type in the label_dict
        self.reg_exp = re.compile(r"5core_xb_pn_\w+_run.npy")
        self.data_files = [f for f in os.listdir(data_folder) if self.reg_exp.match(f)]
        self.cache_data()
    
    def cache_data(self):
        self.data, self.labels = [], []
        for file in self.data_files:
            # extract the modulation type from the file name based on the label_dict
            modualtion_type = file.split("_")[3]
            label = self.label_dict[modualtion_type]
            data = np.load(file)
            
            data = np.mean(data, axis=0)
            label = np.ones(data.shape[0]) * label
            
            self.data.append(data)
            self.labels.append(label)
        
        self.data = np.vstack(self.data)
        self.labels = np.hstack(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.from_numpy(self.data[index]).float()
        data = data.view(1, 1, -1) if self.with_CNN else data
        label = torch.from_numpy(self.labels[index]).long()
        return data, label

    @staticmethod
    def collate_fn(batch):
        data, label = list(zip(*batch))
        data = torch.stack(data)
        label = torch.stack(label)
        return data, label

def stratified_split(dataset, train_ratio=0.8):
    label_dict = defaultdict(list)
    for i in range(len(dataset)):
        _, target = dataset[i]
        label_dict[target.item()].append(i)
        
    train_indices = []
    val_indices = []
    for labels, indices in label_dict.items():
        np.random.shuffle(indices)
        split = int(len(indices) * train_ratio)
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])
        
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    return train_dataset, val_dataset
        
class Cls_model_with_CNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(Cls_model_with_CNN, self).__init__()
        self.cnn = nn.Conv2d(1, 1, (1, 3), 1, (0, 1))
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 128)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x)
        x = self.flat(x)
        x = self.fc3(self.act2(self.fc2(self.act1(self.fc1(x)))))
        return x

class Cls_model_without_CNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(Cls_model_without_CNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc3(self.act2(self.fc2(self.act1(self.fc1(x)))))
        return x


def train_one_epoch(model: nn.Module, train_loader: torch.utils.data.DataLoader,
                    criterion: nn.Module, optimizer: torch.optim.Optimizer, 
                    device: torch.device, epoch: int):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr')
    metric_logger.add_meter('loss')
    metric_logger.add_meter('acc')

    header = 'Epoch: [{}]'.format(epoch)
    all_preds, all_labels = [], []
    for data, target in metric_logger.log_every(train_loader, 10, header):
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        loss = criterion(output, target)
        acc = (output.argmax(-1) == target).float().mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc=acc.item())
        
        all_preds.append(output.argmax(-1).detach().cpu().numpy())
        all_labels.append(target.detach().cpu().numpy())
    
    all_preds, all_labels = np.hstack(all_preds), np.hstack(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    df_cm = pd.DataFrame(cm, index = ["pam2", "pam4", "psk4", "psk8", "qam4", "qam16", "qam64"])
    plt.figure(figsize = (10,7))
    fig = sn.heatmap(df_cm, annot=True).get_figure()
    plt.close()
        
    metric_logger.synchronize_between_processes()
    return metric_logger['loss'].global_avg, metric_logger['acc'].global_avg, fig

def evaluate(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch,
             criterion: nn.Module, epoch: int):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    all_preds, all_labels = [], []
    for data, target in metric_logger.log_every(data_loader, 10, header):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        
        loss = criterion(output, target)
        acc = (output.argmax(-1) == target).float().mean()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc=acc.item())
        
        all_preds.append(output.argmax(-1).detach().cpu().numpy())
        all_labels.append(target.detach().cpu().numpy())
    
    all_preds, all_labels = np.hstack(all_preds), np.hstack(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    df_cm = pd.DataFrame(cm, index = ["pam2", "pam4", "psk4", "psk8", "qam4", "qam16", "qam64"])
    plt.figure(figsize = (10,7))
    fig = sn.heatmap(df_cm, annot=True).get_figure()
    plt.close()
    
    metric_logger.synchronize_between_processes()
    return metric_logger['loss'].global_avg, metric_logger['acc'].global_avg, fig

def main(args):
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    
    device = torch.device(args.device)
    
    print("Creating data loaders")
    whole_dataset = Cls_dataset(data_folder=args.data_folder, with_CNN=args.with_CNN)
    train_dataset, test_dataset = stratified_split(whole_dataset, 0.8)
    
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                               sampler=train_sampler, collate_fn=whole_dataset.collate_fn,
                                               num_workers=args.num_workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                             sampler=test_sampler, collate_fn=whole_dataset.collate_fn,
                                             num_workers=args.num_workers)
    
    print("Creating model")
    model = Cls_model_with_CNN(whole_dataset.data.shape[1], len(whole_dataset.label_dict)) if args.with_CNN else \
            Cls_model_without_CNN(whole_dataset.data.shape[1], len(whole_dataset.label_dict))
    model.to(device)
    if args.with_CNN:
        tb_writer.add_graph(model, torch.randn(1, 1, 1, whole_dataset.data.shape[1]).to(device))
    else:
        tb_writer.add_graph(model, torch.randn(1, whole_dataset.data.shape[1]).to(device))
    
    num_params, num_layers = sum(p.numel() for p in model.parameters()), len(list(model.parameters()))
    print(f"Number of parameters: {num_params}, Number of layers: {num_layers}")
    
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = 0  # do not move
    
    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc, train_cfm = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()
        val_loss, val_acc, val_cfm = evaluate(model, val_loader, device, criterion, epoch)
        
        tags = ["train/loss", "train/acc", "val/loss", "val/acc"]
        values = [train_loss, train_acc, val_loss, val_acc]
        for tag, value in zip(tags, values):
            tb_writer.add_scalar(tag, value, epoch)
        tb_writer.add_figure("train/confusion_matrix", train_cfm, epoch)
        tb_writer.add_figure("val/confusion_matrix", val_cfm, epoch)
        plt.close(train_cfm)
        plt.close(val_cfm)
        
        if args.output_dir:
            save_file = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "lr_scheduler": scheduler.state_dict(),
            }
            digits = len(str(args.epochs))
            torch.save(save_file, os.path.join(args.output_dir, 'model_{}.pth'.format(str(epoch).zfill(digits))))
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    # export final model to onnx
    dummy_input = torch.randn(1, 1, 1, whole_dataset.data.shape[1]).to(device) if args.with_CNN else \
                  torch.randn(1, whole_dataset.data.shape[1]).to(device)
    model.eval()
    torch.onnx.export(model, dummy_input, os.path.join(args.output_dir, 'model.onnx'),
                      export_params=True, opset_version=11)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data_path', default='./fir_results/', help='dataset')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    
    parser.add_argument('--use_rgb', default=False, type=bool, help='use MF')
    parser.add_argument('--use_HVI', default=True, type=bool, help='use HVI')
    
    parser.add_argument('--job_type', default='recover_rgb', help='job type, cls or recover_rgb')

    parser.add_argument('--device', default='cuda', help='device')

    parser.add_argument('--num-classes', default=10, type=int, help='num_classes')

    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')

    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')

    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--sync_bn', type=bool, default=False, help='whether using SyncBatchNorm')

    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate')

    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--print-freq', default=5, type=int, help='print frequency')

    parser.add_argument('--output-dir', default='./weights/recover', help='path where to save')

    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument('--world-size', default=8, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    if args.output_dir:
        mkdir(args.output_dir)

    main(args)

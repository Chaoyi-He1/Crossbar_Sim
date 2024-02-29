import os
import scipy.io as sio
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import random_split
from typing import Tuple
from tqdm import tqdm

from typing import Optional, Tuple, List, Iterable
from torch import nn, Tensor
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import sys
import torch.distributed as dist
import glob
import os
import random
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from contextlib import contextmanager
from torch.utils.tensorboard import SummaryWriter
import scipy.io
import matplotlib.pyplot as plt


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class Step_Prediction(object):
    def __init__(self, num_steps: int = 12) -> None:
        self.num_steps = num_steps
        self.acc_vec = np.zeros(num_steps)
        self.update_count = 0

    def update(self, steps_acc):
        for i, acc in enumerate(steps_acc):
            if isinstance(acc, torch.Tensor):
                acc = acc.item()
            self.acc_vec[i] += acc
        self.update_count += 1

    def reset(self):
        self.acc_vec = np.zeros(self.num_steps)

    def compute(self):
        return self.acc_vec / self.update_count

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.acc_vec)

    def __str__(self):
        steps_acc = self.compute()
        return " ".join([f"step_{i}: {acc:.4f}" for i, acc in enumerate(steps_acc)])


class XB_dataset_npy(Dataset):
    def __init__(self, out_I_paths, ideal_out_I_paths, mtx_paths, float_mtx_path) -> None:
        super().__init__()
        self.ideal_out_I_path = ideal_out_I_paths
        self.out_I_path = out_I_paths
        self.mtx_path = mtx_paths
        self.float_mtx_path = float_mtx_path
        self.cache_data()
        self.num_in = self.out_I.shape[0]
        # print(self.Weight_mtx.shape[0], self.num_in)

    def cache_data(self):
        num_rows = [np.load(file).shape[0] for file in self.out_I_path]
        self.out_I = np.vstack([np.load(file) for file in self.out_I_path])
        self.ideal_out_I = np.vstack([np.load(file) for file in self.ideal_out_I_path])
        self.Weight_mtx = np.vstack(
            [np.tile(np.load(file), (num_rows[i], 1, 1)) for i, file in enumerate(self.mtx_path)])
        self.float_weight_mtx = np.load(self.float_mtx_path)

    def __len__(self):
        return self.num_in

    def __getitem__(self, index):
        Matrix = np.stack([self.Weight_mtx[index], self.float_weight_mtx], 0)
        Iout = self.out_I[index]
        ideal_Iout = self.ideal_out_I[index]
        return Matrix, Iout, ideal_Iout

    @staticmethod
    def collate_fn(batch):
        Matrix, Iout, ideal_Iout = list(zip(*batch))
        Matrix = torch.from_numpy(np.stack(Matrix, 0)).float()
        Iout = torch.from_numpy(np.stack(Iout, 0)).float()
        ideal_Iout = torch.from_numpy(np.stack(ideal_Iout, 0)).long()
        return Matrix, Iout, ideal_Iout


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


class Calib_model(nn.Module):
    def __init__(self, ctx_dim, out_type) -> None:
        super().__init__()
        self.ctx_dim = ctx_dim
        self.conv = nn.Conv2d(in_channels=2, out_channels=ctx_dim,
                              kernel_size=(16, 32))
        self.MLP = nn.ModuleList([
            ConcatSquashLinear(32, 128, ctx_dim)
        ])
        self.MLP.extend([
                            ConcatSquashLinear(128, 128, ctx_dim)
                        ] * 4)
        self.cls_head = nn.ModuleList([ConcatSquashLinear(128, 256, ctx_dim)] * 32) if out_type == "cls" else \
            ConcatSquashLinear(128, 32, ctx_dim)
        self.out_type = out_type

    def forward(self, mtx, exp_Iout):
        ctx = self.conv(mtx).squeeze()
        ideal_Iout = exp_Iout
        for layer in self.MLP:
            ideal_Iout = layer(ctx, ideal_Iout)
        pred = torch.stack([layer(ctx, ideal_Iout) for layer in self.cls_head], dim=1).to(
            mtx.device) if self.out_type == "cls" else \
            self.cls_head(ctx, ideal_Iout)
        return pred


def accuracy_cal(pred, target):
    pred_cls = torch.argmax(pred, dim=-1) if len(pred.shape) == 3 else \
        torch.round(pred).long()
    # Calculate correct predictions
    correct_predictions = (pred_cls == target)
    # Calculate accuracy for each batch
    accuracies = correct_predictions.float().mean(dim=1)
    return accuracies


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, out_type: str,
                    device: torch.device, epoch: int, max_norm: float = 0.1,
                    ):
    model.train()
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=10, fmt='{value:.6f}'))
    metric_logger.add_meter('accuracy', SmoothedValue(window_size=10, fmt='{value:.6f}'))
    batch_Predict = Step_Prediction(num_steps=32)
    header = 'Epoch: [{}]'.format(epoch)

    for i, (Matrix, Iout, ideal_Iout) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        Matrix = Matrix.to(device)
        Iout = Iout.to(device)
        ideal_Iout = ideal_Iout.to(device)

        I_pred = model(Matrix, Iout)
        loss = F.cross_entropy(I_pred.transpose(1, 2), ideal_Iout) if out_type == 'cls' else \
            F.mse_loss(I_pred, ideal_Iout)
        acc = accuracy_cal(I_pred, ideal_Iout)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        metric_logger.update(accuracy=acc.mean().item())
        batch_Predict.update(acc)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, batch_Predict


@torch.no_grad()
def evaluation(model: torch.nn.Module, data_loader: Iterable,
               out_type: str, device: torch.device, epoch: int,
               ):
    model.eval()
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('accuracy', SmoothedValue(window_size=10, fmt='{value:.6f}'))
    batch_Predict = Step_Prediction(num_steps=32)
    header = 'Test:'

    for i, (Matrix, Iout, ideal_Iout) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        Matrix = Matrix.to(device)
        Iout = Iout.to(device)
        ideal_Iout = ideal_Iout.to(device)

        I_pred = model(Matrix, Iout)
        loss = F.cross_entropy(I_pred.transpose(1, 2), ideal_Iout) if out_type == 'cls' else \
            F.mse_loss(I_pred, ideal_Iout)
        acc = accuracy_cal(I_pred, ideal_Iout)

        metric_logger.update(loss=loss.item())
        metric_logger.update(accuracy=acc.mean().item())
        batch_Predict.update(acc)

    print("Averaged stats metric logger:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, batch_Predict


def main():
    float_mtx_path = '../data/Calib/uniform_weight.npy'
    mtx_path = ['../data/Calib/qtz_mtx_xb_uniform.npy',
                '../data/Calib/qtz_mtx_xb_uniform_bit_norm.npy',
                '../data/Calib/qtz_mtx_xb_uniform_mid_g.npy']
    ideal_out_I_path = ['../data/Calib/ideal_out_xb_uniform.npy',
                        '../data/Calib/ideal_out_xb_uniform_bit_norm.npy',
                        '../data/Calib/ideal_out_xb_uniform_mid_g.npy']
    out_I_path = ['../data/Calib/xb_out_xb_uniform.npy',
                  '../data/Calib/xb_out_xb_uniform_bit_norm.npy',
                  '../data/Calib/xb_out_xb_uniform_mid_g.npy']

    device = torch.device('cuda')
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")
    batch_size = 32
    seed = 42
    epochs = 50
    lr = 1e-4
    lrf = 0.01
    reg_lambda = 0.6
    out_type = 'mse'  # mse or cls

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tb_writer = SummaryWriter("../runs/")

    print("MNIST dataset generating...")
    dataset = XB_dataset_npy(out_I_paths=out_I_path,
                             ideal_out_I_paths=ideal_out_I_path,
                             mtx_paths=mtx_path,
                             float_mtx_path=float_mtx_path)
    dataset_train, dataset_val = random_split(dataset,
                                              [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, batch_size, drop_last=True)
    batch_sampler_val = torch.utils.data.BatchSampler(sampler_val, batch_size, drop_last=True)

    print("MNIST dataloader generating...")
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using %g dataloader workers' % nw)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                                    collate_fn=dataset.collate_fn, num_workers=nw)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_sampler=batch_sampler_val,
                                                  collate_fn=dataset.collate_fn, num_workers=nw)

    print("Model generating...")
    model = Calib_model(256, out_type)
    model.to(device)

    tb_writer.add_graph(model, (torch.randn((batch_size, 2, 16, 32),
                                            device=device, dtype=torch.float),
                                torch.randn((batch_size, 32),
                                            device=device, dtype=torch.float)),
                        use_strict_trace=False)

    start_epoch = 0
    params_to_optimize = []
    n_parameters, layers = 0, 0
    for p in model.parameters():
        n_parameters += p.numel()
        layers += 1
        if p.requires_grad:
            params_to_optimize.append(p)
    print('number of params:', n_parameters)
    print('Model Summary: %g layers, %g parameters' % (layers, n_parameters))
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch

    # criterion = nn.BCEWithLogitsLoss()

    print("Start training...")
    start_time = time.time()

    for epoch in range(start_epoch, epochs + start_epoch):
        train_loss_dict, step_acc = train_one_epoch(model=model, data_loader=data_loader_train,
                                                    optimizer=optimizer, device=device, epoch=epoch,
                                                    max_norm=0.1, out_type=out_type)
        print(str(step_acc))
        scheduler.step()

        test_loss_dict, step_acc = evaluation(model=model, data_loader=data_loader_val,
                                              device=device, epoch=epoch, out_type=out_type)
        print(str(step_acc))

        items = {
            **{f'train_{k}': v for k, v in train_loss_dict.items()},
            **{f'test_{k}': v for k, v in test_loss_dict.items()},
        }
        for k, v in items.items():
            tb_writer.add_scalar(k, v, epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # save model
    digits = len(str(epochs))
    torch.save({"epoch": epochs,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                },
               '../weights/model_{}.pth'.format(str(epochs).zfill(digits)))


if __name__ == "__main__":
    main()

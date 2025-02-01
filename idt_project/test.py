import time
import os
import datetime
import random
import math
import torch
import copy
from pathlib import Path

import torch.utils
import torch.utils.data
from model import CNN_BN, CNN_conv, mlp_model
from train_eval import train_one_epoch, evaluate
from datasets import idt_dataset, custom_random_sampler, idt_dataset_mlp
import misc
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import subprocess

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='IDT Project')
    parser.add_argument('--train_data', default='/data/chaoyi_he/Crossbar_Sim/idt_project/data/Train/train_features.npy', type=str)
    parser.add_argument('--train_label', default='/data/chaoyi_he/Crossbar_Sim/idt_project/data/Train/train_labels.npy', type=str)
    parser.add_argument('--test_data', default='/data/chaoyi_he/Crossbar_Sim/idt_project/data/Test/val_features.npy', type=str)
    parser.add_argument('--test_label', default='/data/chaoyi_he/Crossbar_Sim/idt_project/data/Test/val_labels.npy', type=str)
    
    parser.add_argument('--model', default='mlp_model', type=str)
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--num_classes', default=30, type=int)
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lrf', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.2, type=float)
    
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--output-dir', default='./weights/', help='path where to save')
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    return parser.parse_args()


def get_gpu_power(gpu_id=0):
    result = subprocess.run(
        ['nvidia-smi', f'--query-gpu=power.draw', '--format=csv,noheader,nounits', f'--id={gpu_id}'],
        stdout=subprocess.PIPE
    )
    return float(result.stdout.decode('utf-8').strip())


def main(args):
    print(args)
    device = torch.device(args.device)
    
    print("Creating data loaders")
    train_dataset = idt_dataset_mlp(args.train_data, args.train_label)
    val_dataset = idt_dataset_mlp(args.test_data, args.test_label)
    
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                               sampler=train_sampler, num_workers=args.num_workers,
                                               drop_last=True, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             sampler=val_sampler, num_workers=args.num_workers,
                                             drop_last=True, collate_fn=val_dataset.collate_fn)
    
    print("Creating model: {}".format(args.model))
    if args.model == 'CNN_BN':
        model = CNN_BN(1, args.num_classes, train_dataset.h, train_dataset.w)
    elif args.model == 'CNN_conv':
        model = CNN_conv(1, args.num_classes, train_dataset.h, train_dataset.w)
    elif args.model == 'mlp_model':
        model = mlp_model(64, args.num_classes)
    else:
        raise ValueError("Model not supported")
    model.to(device)
    
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    checkpoint = torch.load('weights/model_best_50.pth')
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    
    TDP = 280
    
    print("Start testing")
    # record the testing time in seconds and power consumption in watts
    start_time = time.time()
    # power consumption
    initial_cpu_times = psutil.cpu_times_percent(interval=None)
    start_power = get_gpu_power()
    
    val_loss, val_acc, val_cfm = evaluate(model, train_loader, device, 0, args.print_freq, scaler, args.num_classes)
    val_loss, val_acc, val_cfm = evaluate(model, val_loader, device, 0, args.print_freq, scaler, args.num_classes)
    
    # record the testing time in seconds
    end_time = time.time()
    end_power = get_gpu_power()
    final_cpu_times = psutil.cpu_times_percent(interval=None)
    test_time = end_time - start_time
    print("Testing time: ", test_time)
    
    cpu_usage_percent = (
        (final_cpu_times.user - initial_cpu_times.user) +
        (final_cpu_times.system - initial_cpu_times.system)
    ) / (end_time - start_time)
    
    # Estimate average power consumption during inference
    power_usage = TDP * (cpu_usage_percent / 100)
    # Calculate energy consumed (Power * Time in seconds)
    energy_consumed = power_usage * (end_time - start_time)

    print(f'Estimated Energy Consumed: {energy_consumed:.2f} Joules')
    # Calculate average power and energy consumed
    avg_power = (start_power + end_power) / 2
    elapsed_time = end_time - start_time
    energy_consumed = avg_power * elapsed_time  # Energy in watt-seconds (Joules)

    print(f'Elapsed time: {elapsed_time:.2f} seconds')
    print(f'Estimated Energy Consumed: {energy_consumed:.2f} Joules')
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
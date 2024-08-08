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
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lrf', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.01, type=float)
    
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

def main(args):
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    
    device = torch.device(args.device)
    
    # fix the seed
    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    
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
    
    if args.model == 'CNN_BN':
        tb_writer.add_graph(model, torch.randn(1, 1, train_dataset.h, train_dataset.w).to(device))
    elif args.model == 'CNN_conv':
        tb_writer.add_graph(model, torch.randn(1, 1, train_dataset.h, train_dataset.w).to(device))
    elif args.model == 'mlp_model':
        tb_writer.add_graph(model, torch.randn(1, 1, 8, 8).to(device))
        
    num_params, num_layers = sum(p.numel() for p in model.parameters()), len(list(model.parameters()))
    print("Number of parameters: {}".format(num_params), "Number of layers: {}".format(num_layers))
    
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr)
    
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = 0
    
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        # before each epoch, add gaussian noise to the model parameters
        # for p in model.parameters():
        #     if p.requires_grad:
        #         p.data.add_(torch.randn_like(p) * 0.01)
        
        if epoch <= 5 or (epoch > 10 and epoch <= 15) or (epoch > 20 and epoch <= 25) or (epoch > 30 and epoch <= 35) or (epoch > 40 and epoch <= 47):
            train_loss, train_acc, train_cfm = train_one_epoch(model, optimizer, args.alpha, train_loader, device, epoch, 
                                                            args.print_freq, scaler, args.num_classes)
            
            val_loss, val_acc, val_cfm = evaluate(model, val_loader, device, epoch, args.print_freq, scaler, args.num_classes)
        elif (epoch > 5 and epoch <= 10) or (epoch > 15 and epoch <= 20) or (epoch > 25 and epoch <= 30) or (epoch > 35 and epoch <= 40) or (epoch > 47 and epoch <= 50):
            train_loss, train_acc, train_cfm = train_one_epoch(model, optimizer, args.alpha, val_loader, device, epoch, 
                                                            args.print_freq, scaler, args.num_classes)
            
            val_loss, val_acc, val_cfm = evaluate(model, train_loader, device, epoch, args.print_freq, scaler, args.num_classes)
        
        # train_loss, train_acc, train_cfm = train_one_epoch(model, optimizer, args.alpha, train_loader, device, epoch, 
        #                                                    args.print_freq, scaler, args.num_classes)
        
        # val_loss, val_acc, val_cfm = evaluate(model, val_loader, device, epoch, args.print_freq, scaler, args.num_classes)
        
        scheduler.step()
        
        # Save the best model
        save_file = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            "epoch": epoch,
        }
        digits = len(str(args.epochs))
        torch.save(save_file, os.path.join(args.output_dir, 'model_best_{}.pth'.format(str(epoch).zfill(digits))))
        
        tb_writer.add_scalar("train/loss", train_loss, epoch)
        tb_writer.add_scalar("train/acc", train_acc, epoch)
        tb_writer.add_scalar("val/loss", val_loss, epoch)
        tb_writer.add_scalar("val/acc", val_acc, epoch)
        tb_writer.add_figure("train/Confusion Matrix", train_cfm, epoch)
        tb_writer.add_figure("val/Confusion Matrix", val_cfm, epoch)
        plt.close(train_cfm)
        plt.close(val_cfm)
    
    print("Training time: ", datetime.timedelta(seconds=time.time() - start_time))
    tb_writer.close()
    
    # export model to onnx
    if args.model == 'CNN_BN' or args.model == 'CNN_conv':
        dummy_input = torch.randn(1, 1, train_dataset.h, train_dataset.w).to(device)
    elif args.model == 'mlp_model':
        dummy_input = torch.randn(1, 1, 8, 8).to(device)
    torch.onnx.export(model, dummy_input, 
                      os.path.join(args.output_dir, 'model_best.onnx'), 
                      export_params=True, opset_version=16, dynamic_axes={'input': {0: 'batch_size'}, 'sensor_out': {0: 'batch_size'}},
                      input_names=['input'], output_names=['sensor_out'])
    
    # min, max of the model parameters
    min_params, max_params = [], []
    for p in model.parameters():
        min_params.append(p.min().item())
        max_params.append(p.max().item())
    min_params, max_params = np.min(min_params), np.max(max_params)
    print(f"Min of model parameters: {min_params}, Max of model parameters: {max_params}")
    # plot histogram of model parameters
    params = [p.detach().cpu().numpy().flatten() for p in model.parameters()]
    plt.hist(np.hstack(params), bins=100)
    plt.savefig(os.path.join(args.output_dir, 'model_params_hist.png'))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
        
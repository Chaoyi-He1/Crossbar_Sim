import time
import os
import datetime
import random
import math
import torch
import copy
from pathlib import Path
from model import CNN_BN_bone, CNN_conv_bone, mlp_model, Conv2d_AutoEncoder
from train_eval.train_contrast import train_one_epoch
from train_eval.eval_contrast import evaluate
from datasets import idt_dataset, custom_random_sampler
import misc
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='IDT Project')
    parser.add_argument('--train_data', default='/data/chaoyi_he/Crossbar_Sim/idt_project/data/Train/idt_train_data.npy', type=str)
    parser.add_argument('--train_label', default='/data/chaoyi_he/Crossbar_Sim/idt_project/data/Train/idt_train_label.npy', type=str)
    parser.add_argument('--test_data', default='/data/chaoyi_he/Crossbar_Sim/idt_project/data/Test/idt_test_data.npy', type=str)
    parser.add_argument('--test_label', default='/data/chaoyi_he/Crossbar_Sim/idt_project/data/Test/idt_test_label.npy', type=str)
    
    parser.add_argument('--model', default='ResNet', type=str)
    parser.add_argument('-b', '--batch-size', default=30, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--num_classes', default=30, type=int)
    parser.add_argument('--epochs', default=170, type=int)
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
    train_dataset = idt_dataset(args.train_data, args.train_label)
    val_dataset = idt_dataset(args.test_data, args.test_label)
    
    train_sampler = custom_random_sampler(train_dataset, args.batch_size)
    val_sampler = custom_random_sampler(val_dataset, args.batch_size)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                               sampler=train_sampler, num_workers=args.num_workers,
                                               drop_last=True, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             sampler=val_sampler, num_workers=args.num_workers,
                                             drop_last=True, collate_fn=val_dataset.collate_fn)
    
    print("Creating model: {}".format(args.model))
    if args.model == 'CNN_BN':
        model = CNN_BN_bone(1, args.num_classes, train_dataset.h, train_dataset.w)
    elif args.model == 'CNN_conv':
        model = CNN_conv_bone(1, args.num_classes, train_dataset.h, train_dataset.w)
    elif args.model == 'mlp_model':
        model = mlp_model(train_dataset.h * train_dataset.w, args.num_classes)
    elif args.model == 'ResNet':
        model = Conv2d_AutoEncoder(in_dim=(train_dataset.h, train_dataset.w), in_channel=1)
    else:
        raise ValueError("Model not supported")
    model.to(device)
    
    if args.model == 'CNN_BN':
        tb_writer.add_graph(model, torch.randn(1, 1, train_dataset.h, train_dataset.w).to(device))
    elif args.model == 'CNN_conv':
        tb_writer.add_graph(model, torch.randn(1, 1, train_dataset.h, train_dataset.w).to(device))
    elif args.model == 'mlp_model':
        tb_writer.add_graph(model, torch.randn(1, train_dataset.h * train_dataset.w).to(device))
    elif args.model == 'ResNet':
        tb_writer.add_graph(model, torch.randn(1, 1, train_dataset.h, train_dataset.w).to(device))
        
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
        train_loss = train_one_epoch(model, optimizer, args.alpha, train_loader, device, epoch, 
                                     args.print_freq, scaler, args.num_classes)
        if epoch % 50 == 0:
            val_loss, val_t_sne = evaluate(model, val_loader, device, epoch, args.print_freq, scaler, args.num_classes)
        else:
            val_loss = evaluate(model, val_loader, device, epoch, args.print_freq, scaler, args.num_classes)
        
        scheduler.step()
        
        tb_writer.add_scalar("train/loss", train_loss, epoch)
        tb_writer.add_scalar("val/loss", val_loss, epoch)
        if epoch % 50 == 0:
            tb_writer.add_figure("val/t-SNE", val_t_sne, epoch)
            plt.close(val_t_sne)
        
        # Save the best model
        save_file = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            "epoch": epoch,
        }
        digits = len(str(args.epochs))
        torch.save(save_file, os.path.join(args.output_dir, 'model_{}.pth'.format(str(epoch).zfill(digits))))
    
    print("Training time: ", datetime.timedelta(seconds=time.time() - start_time))
    tb_writer.close()
    
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
        
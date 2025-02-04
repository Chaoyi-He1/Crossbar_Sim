import time
import os
import datetime
import random
import math
import torch
import copy
from pathlib import Path
from model import CNN_BN, CNN_conv, mlp_model, AutoEncoder_cls, mlp_encoder
from train_eval import train_one_epoch, evaluate
from datasets import *
import misc
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='IDT Project')
    parser.add_argument('--train_data', default='/data/chaoyi_he/Crossbar_Sim/idt_project/data/Train/idt_train_data_wifi_whole.npy', type=str)
    parser.add_argument('--train_label', default='/data/chaoyi_he/Crossbar_Sim/idt_project/data/Train/idt_train_label_wifi_whole.npy', type=str)
    parser.add_argument('--test_data', default='/data/chaoyi_he/Crossbar_Sim/idt_project/data/Test/idt_test_data_wifi.npy', type=str)
    parser.add_argument('--test_label', default='/data/chaoyi_he/Crossbar_Sim/idt_project/data/Test/idt_test_label_wifi.npy', type=str)
    parser.add_argument('--resume', default='/data/chaoyi_he/Crossbar_Sim/weights/wifi/model_09', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    
    parser.add_argument('--model', default='CNN_conv', type=str)
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--num_classes', default=31, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lrf', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.0, type=float)
    
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=1000, type=int, help='print frequency')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--output-dir', default='./weights/wifi/', help='path where to save')
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
    train_dataset = idt_dataset_wifi_anomaly(args.train_data, args.train_label)
    val_dataset = idt_dataset_wifi_anomaly(args.test_data, args.test_label)
    
    # train_sampler = custom_random_sampler(train_dataset, args.batch_size)
    # val_sampler = custom_random_sampler(val_dataset, args.batch_size)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.RandomSampler(val_dataset)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                               sampler=train_sampler, num_workers=args.num_workers,
                                               drop_last=False, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             sampler=val_sampler, num_workers=args.num_workers,
                                             drop_last=False, collate_fn=val_dataset.collate_fn)
    
    # check if every data in val_dataset is in train_dataset
    '''
    data in test_data is shape (N, 16, 16, 2)
    train_data is shape (M, 16, 16, 2)
    check if every (16, 16, 2) in test_data is in train_data
    and check if the label is the same
    '''
    # for i in range(len(val_dataset)):
    #     if not np.any(np.all(val_dataset.data[i] == train_dataset.data, axis=(1, 2, 3))):
    #         print("test data not in train data: {}".format(i))
    #         break
    #     else:
    #         if train_dataset.label[np.argmax(np.all(val_dataset.data[i] == train_dataset.data, axis=(1, 2, 3)))] != val_dataset[i][1]:
    #             print("test label not in train label")
    #             break
    # print("All test data in train data") 
    
    print("Creating model: {}".format(args.model))
    if args.model == 'CNN_BN':
        model = CNN_BN(1, args.num_classes, train_dataset.h, train_dataset.w)
    elif args.model == 'CNN_conv':
        model = CNN_conv(2, args.num_classes, train_dataset.h, train_dataset.w)
    elif args.model == 'mlp_model':
        model = mlp_encoder(2 * train_dataset.h * train_dataset.w, args.num_classes)
    elif args.model == 'ResNet':
        model = AutoEncoder_cls(in_dim=(train_dataset.h, train_dataset.w), in_channel=1, num_cls=args.num_classes)
    else:
        raise ValueError("Model not supported")
    model.to(device)
    
    start_epoch = 0
    if args.resume.endswith('.pth'):
        print("Loading checkpoint: {}".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
    
    if args.model == 'CNN_BN':
        tb_writer.add_graph(model, torch.randn(1, 1, train_dataset.h, train_dataset.w).to(device))
    elif args.model == 'CNN_conv':
        tb_writer.add_graph(model, torch.randn(1, 2, train_dataset.h, train_dataset.w).to(device))
    elif args.model == 'mlp_model':
        tb_writer.add_graph(model, torch.randn(1, 2 * train_dataset.h * train_dataset.w).to(device))
    elif args.model == 'ResNet':
        tb_writer.add_graph(model, torch.randn(1, 1, train_dataset.h, train_dataset.w).to(device))
        
    num_params, num_layers = sum(p.numel() for p in model.parameters()), len(list(model.parameters()))
    print("Number of parameters: {}".format(num_params), "Number of layers: {}".format(num_layers))
    
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr)
    
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch
    
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + start_epoch):
        # train_loss = train_one_epoch(model, optimizer, args.alpha, train_loader, device, epoch, 
        #                              args.print_freq, scaler, args.num_classes)
        # if epoch % 50 == 0:
        #     val_loss, val_t_sne = evaluate(model, val_loader, device, epoch, args.print_freq, scaler, args.num_classes)
        # else:
        #     val_loss = evaluate(model, val_loader, device, epoch, args.print_freq, scaler, args.num_classes)
        
        train_loss, train_acc, train_cfm = train_one_epoch(model, optimizer, args.alpha, train_loader, device, epoch, 
                                                           args.print_freq, scaler, args.num_classes)
        
        val_loss, val_acc, val_cfm = evaluate(model, val_loader, device, epoch, args.print_freq, scaler, args.num_classes)
        scheduler.step()
        
        tb_writer.add_scalar("train/loss", train_loss, epoch)
        tb_writer.add_scalar("train/acc", train_acc, epoch)
        tb_writer.add_scalar("val/loss", val_loss, epoch)
        tb_writer.add_scalar("val/acc", val_acc, epoch)
        tb_writer.add_figure("train/Confusion Matrix", train_cfm, epoch)
        tb_writer.add_figure("val/Confusion Matrix", val_cfm, epoch)
        # plt.close(train_cfm)
        # plt.close(val_cfm)
        # if epoch % 50 == 0:
        #     tb_writer.add_figure("val/t-SNE", val_t_sne, epoch)
        #     plt.close(val_t_sne)
        
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
        
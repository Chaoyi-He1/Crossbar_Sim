import time
import os
import datetime
import random
import math
import torch
import copy
from pathlib import Path
from model import CNN_BN, CNN_conv, mlp_model, AutoEncoder_cls, CNN_conv_bone
from train_eval import train_one_epoch, evaluate, feature_extractor
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
    
    parser.add_argument('--resume', default='/data/chaoyi_he/Crossbar_Sim/weights/model_best_199.pth', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    
    parser.add_argument('--model', default='CNN_conv', type=str)
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--num_classes', default=30, type=int)
    parser.add_argument('--epochs', default=100, type=int)
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
    
    print("Creating model: {}".format(args.model))
    if args.model == 'CNN_BN':
        model = CNN_BN(1, args.num_classes, train_dataset.h, train_dataset.w)
    elif args.model == 'CNN_conv':
        model = CNN_conv_bone(1, args.num_classes, train_dataset.h, train_dataset.w)
    elif args.model == 'mlp_model':
        model = mlp_model(train_dataset.h * train_dataset.w, args.num_classes)
    elif args.model == 'ResNet':
        model = AutoEncoder_cls(in_dim=(train_dataset.h, train_dataset.w), in_channel=1, num_cls=args.num_classes)
    else:
        raise ValueError("Model not supported")
    model.to(device)
    
    if args.resume.endswith('.pth'):
        print("Loading checkpoint: {}".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    print("Start latent extractor inference")
    start_time = time.time()
    
    train_t_sne, train_features, train_labels = feature_extractor(model, train_loader, device, 0, args.print_freq, scaler, args.num_classes)
        
    val_t_sne, val_features, val_labels = feature_extractor(model, val_loader, device, 0, args.print_freq, scaler, args.num_classes)
        
    train_t_sne.savefig(os.path.join(args.output_dir, 'train_t_sne.png'))
    val_t_sne.savefig(os.path.join(args.output_dir, 'val_t_sne.png'))
    
    np.save(os.path.join(args.output_dir, 'train_features.npy'), train_features)
    np.save(os.path.join(args.output_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(args.output_dir, 'val_features.npy'), val_features)
    np.save(os.path.join(args.output_dir, 'val_labels.npy'), val_labels)
    
    print("Training time: ", datetime.timedelta(seconds=time.time() - start_time))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
        
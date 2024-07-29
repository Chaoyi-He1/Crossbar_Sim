import numpy as np
import os
import torch.utils
from torch.utils.data import Dataset
import torch.utils.data
from torchvision import transforms
import torch
from collections import defaultdict


def transform_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform


class idt_dataset(Dataset):
    def __init__(self, data_path, label_path):
        super(idt_dataset, self).__init__()
        self.data = np.load(data_path)
        self.label = np.load(label_path)
        self.transform = transform_data()
        self.h, self.w = self.data.shape[1], self.data.shape[2]
        # self.reformat_data()
    
    def reformat_data(self):
        self.data_dict = defaultdict(list)
        for i in range(len(self.label)):
            self.data_dict[self.label[i]].append(self.data[i])
        self.min_len = min([len(self.data_dict[key]) for key in self.data_dict.keys()])
        for k, v in self.data_dict.items():
            if len(v) > self.min_len:
                # Randomly select self.min_len samples
                self.data_dict[k] = np.random.choice(v, self.min_len, replace=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = torch.from_numpy(self.label[idx]).long()
        if self.transform:
            data = self.transform(data)
        return data, label
    
    @staticmethod
    def collate_fn(batch):
        data, label = list(zip(*batch))
        data = torch.stack(data, dim=0)
        label = torch.concat(label, dim=0).view(-1)
        return data, label

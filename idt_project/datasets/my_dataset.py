import numpy as np
import os
import torch.utils
import random
from torch.utils.data import Dataset
import torch.utils.data
from torchvision import transforms
import torch
from collections import defaultdict
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union


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
            self.data_dict[self.label[i, 0]].append(i)
        self.min_len = min([len(self.data_dict[key]) for key in self.data_dict.keys()])
        for k, v in self.data_dict.items():
            if len(v) > self.min_len:
                # Randomly select self.min_len samples
                self.data_dict[k] = random.sample(v, self.min_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = torch.tensor(self.label[idx]).long()
        if self.transform:
            data = self.transform(data)
        return data, label
    
    @staticmethod
    def collate_fn(batch):
        data, label = list(zip(*batch))
        data = torch.stack(data, dim=0)
        label = torch.cat(label, dim=0)
        return data, label


class idt_dataset_wifi(Dataset):
    def __init__(self, data_path, label_path):
        super(idt_dataset_wifi, self).__init__()
        self.data = np.load(data_path)
        self.label = np.load(label_path)
        
        self.data = self.data[self.label != 30]
        self.label = self.label[self.label != 30]
        
        self.transform = transform_data()
        self.h, self.w = self.data.shape[1], self.data.shape[2]
        
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min()) * 230 + 10
        self.data = self.data.astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = torch.tensor(self.label[idx]).long()
        if self.transform:
            data = self.transform(data)
        return data, label
    
    @staticmethod
    def collate_fn(batch):
        data, label = list(zip(*batch))
        data = torch.stack(data, dim=0)
        label = torch.tensor(label)
        return data, label

class idt_dataset_wifi_anomaly(Dataset):
    def __init__(self, data_path, label_path):
        super(idt_dataset_wifi_anomaly, self).__init__()
        self.data = np.load(data_path)
        self.label = np.load(label_path)
        
        self.transform = transform_data()
        self.h, self.w = self.data.shape[1], self.data.shape[2]
        
        # self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min()) * 230 + 10
        # self.data = self.data.astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = torch.tensor(self.label[idx]).long()
        anomaly = torch.tensor(0)
        if self.transform:
            data = self.transform(data)
        if label == 30:
            anomaly = torch.tensor(1)
        return data, label, anomaly
    
    @staticmethod
    def collate_fn(batch):
        data, label, anomaly = list(zip(*batch))
        data = torch.stack(data, dim=0)
        label = torch.tensor(label)
        anomaly = torch.tensor(anomaly)
        return data, label, anomaly
    
class idt_dataset_mlp(Dataset):
    def __init__(self, data_path, label_path):
        super(idt_dataset_mlp, self).__init__()
        self.data = np.load(data_path)
        #rescale the data from [0, 255] to [10, 240] integer values
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min()) * 230 + 10
        self.data = self.data.astype(np.uint8)
        self.label = np.load(label_path)
        self.transform = transform_data()
        self.h, self.w = 8, 8
        # self.reformat_data()
    
    def reformat_data(self):
        self.data_dict = defaultdict(list)
        for i in range(len(self.label)):
            self.data_dict[self.label[i, 0]].append(i)
        self.min_len = min([len(self.data_dict[key]) for key in self.data_dict.keys()])
        for k, v in self.data_dict.items():
            if len(v) > self.min_len:
                # Randomly select self.min_len samples
                self.data_dict[k] = random.sample(v, self.min_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = torch.tensor(self.label[idx]).long()
        if self.transform:
            data = data.reshape(8, 8, 1)
            data = self.transform(data)
        return data, label
    
    @staticmethod
    def collate_fn(batch):
        data, label = list(zip(*batch))
        data = torch.stack(data, dim=0)
        label = torch.stack(label, dim=0)
        return data, label


class custom_random_sampler(torch.utils.data.RandomSampler):
    def __init__(self, data_source: idt_dataset, batch_size: int, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        super().__init__(data_source, replacement, num_samples, generator)
        self.batch_size = batch_size
    
    def __iter__(self) -> Iterator[int]:
        """
        generate the indices for the data, the indices are generated randomly
        but guarantee that for each label the batch contains, there are two samples
        """
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        
        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            self.data_source.reformat_data()
            for _ in range(self.num_samples // self.batch_size):
                # Randomly select batch_size / 2 labels from the total num_labels (len(self.data_source.data_dict)),
                # data_dict is a dictionary with key as the label and value as the list of corresponding samples indices
                # then for each selected label, randomly select two samples, then remove the selected samples from the data_dict
                # and repeat the process until all the samples are selected
                
                #First, sort the self.data_source.data_dict based on the length of the value list, in descending order,
                # if the length of the value lists are the same, randomly shuffle the dictionary order
                
                # check if the length of the value lists are the same
                value_lens = [len(v) for v in self.data_source.data_dict.values()]
                if len(set(value_lens)) == 1:
                    # randomly shuffle the dictionary order
                    data_dict_items = list(self.data_source.data_dict.items())
                    random.shuffle(data_dict_items)
                    self.data_source.data_dict = dict(data_dict_items)
                else:
                    # sort the dictionary based on the length of the value list in descending order
                    data_dict_items = sorted(self.data_source.data_dict.items(), key=lambda x: len(x[1]), reverse=True)
                    self.data_source.data_dict = dict(data_dict_items)

                available_labels = [k for k, v in self.data_source.data_dict.items() if len(v) > 0]
                if len(available_labels) < self.batch_size // 2:
                    break
                selected_labels = available_labels[:self.batch_size // 2]
                indices = []
                for label in selected_labels:
                    selected_indices = random.sample(self.data_source.data_dict[label], 2)
                    for i in selected_indices:
                        indices.append(i)
                        self.data_source.data_dict[label].remove(i)
                yield from indices
            available_labels = [k for k, v in self.data_source.data_dict.items() if len(v) > 0]
            indices = []
            for label in available_labels:
                selected_indices = random.sample(self.data_source.data_dict[label], 2)
                for i in selected_indices:
                    indices.append(i)
                    self.data_source.data_dict[label].remove(i)
            yield from indices

from dataset_preparation import *
import numpy as np
import pandas as pd
import os


def get_raw_data(file_path):
    """
    Read the raw data from the file
    :param file_path: the path of the file
    :return: the raw data
    """
    customized_loader = LoadDataset()
    channel_obj = ChannelIndSpectrogram()
    
    snr_range = np.arange(40,80)
    
    assert os.path.isfile(file_path), "The file does not exist."
    data, label = customized_loader.load_iq_samples(file_path, 
                                                    dev_range=np.arange(0, 30, dtype=int),
                                                    pkt_range=np.arange(0, 1000, dtype=int))
    data = awgn(data, snr_range)
    data = channel_obj.channel_ind_spectrogram(data)
    
    print("The shape of the data is: ", data.shape)
    print("The shape of the label is: ", label.shape)
    print("mean of the data: ", np.mean(data), "std of the data: ", np.std(data))
    
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    data = data.astype(np.uint8)
    
    # for each label, separate the data into two parts, training and testing with 80% and 20% respectively
    train_data, test_data, train_label, test_label = [], [], [], []
    for i in range(30):
        idx = np.where(label == i)[0]
        np.random.shuffle(idx)
        train_data.append(data[idx[:int(0.8 * len(idx))]])
        test_data.append(data[idx[int(0.8 * len(idx)):]])
        train_label.append(label[idx[:int(0.8 * len(idx))]])
        test_label.append(label[idx[int(0.8 * len(idx)):]])
    
    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)
    train_label = np.concatenate(train_label)
    test_label = np.concatenate(test_label)
    
    save_path = "/data/chaoyi_he/Crossbar_Sim/idt_project/data/Test/"
    np.save(save_path+"idt_test_data.npy", test_data)
    np.save(save_path+"idt_test_label.npy", test_label)
    
    save_path = "/data/chaoyi_he/Crossbar_Sim/idt_project/data/Train/"
    np.save(save_path+"idt_train_data.npy", train_data)
    np.save(save_path+"idt_train_label.npy", train_label)

if __name__ == "__main__":
    file_path = "/data/chaoyi_he/Crossbar_Sim/idt_project/data/Train/dataset_training_aug.h5"
    get_raw_data(file_path)
    
    
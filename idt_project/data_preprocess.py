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
                                                    pkt_range=np.arange(0, 500, dtype=int))
    data = awgn(data, snr_range)
    data = channel_obj.channel_ind_spectrogram(data)
    
    print("The shape of the data is: ", data.shape)
    print("The shape of the label is: ", label.shape)
    print("mean of the data: ", np.mean(data), "std of the data: ", np.std(data))
    
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    save_path = "/data/chaoyi_he/Crossbar_Sim/idt_project/data/Train/"
    np.save(save_path+"idt_train_data.npy", data)
    np.save(save_path+"idt_train_label.npy", label)
    
    return data, label

if __name__ == "__main__":
    file_path = "/data/chaoyi_he/Crossbar_Sim/idt_project/data/Train/dataset_training_aug.h5"
    data, label = get_raw_data(file_path)
    
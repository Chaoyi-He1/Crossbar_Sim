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
    
    return data, label


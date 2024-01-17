import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from python.VMM_post_process import *


'''
This function creates a DFT matrix of size N x M,
where N is the number of samples and M is the number of frequencies.
The DFT matrix will be separated into real and imaginary parts.
The ourput matrix will be of size 2N x 2M, 
where the [0:N, 0:M] matrix is the real part, and the [N:2N, M:2M] matrix is the imaginary part.
'''
def DFT_mtx_create(N, M):
    n = np.arange(N)
    m = np.arange(M)
    nm = np.outer(n, m)
    DFT_mtx = np.exp(-2j * np.pi * nm / N)
    
    # Split the DFT matrix into real and imaginary parts,
    # and stack them together to form a 2N x 2M matrix.
    DFT_mtx_out = np.hstack((np.real(DFT_mtx), np.zeros((N, M))))
    DFT_mtx_out = np.vstack((DFT_mtx_out, np.hstack((np.zeros((N, M)), np.imag(DFT_mtx)))))
    
    return DFT_mtx_out


if __name__ == '__main__':
    # Read the data from the VMM post-processed data file.
    # The data is in the form of a pandas DataFrame.
    data = pd.read_csv('data/VMM_post_processed_data.csv')
    data = data.to_numpy()
    
    # Extract the data fro
import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from VMM_sim import *
from VMM_post_process import *
from visualize import plot_array

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


def complex_vmm_to_vmm(input, weight):
    no_batch, in_dim = input.shape
    in_whole = np.zeros((2*no_batch, 2*in_dim))
    in_whole[0:no_batch, 0:in_dim] = np.real(input)
    in_whole[0:no_batch, in_dim:2*in_dim] = np.real(input)
    in_whole[no_batch:2*no_batch, 0:in_dim] = np.imag(input)
    in_whole[no_batch:2*no_batch, in_dim:2*in_dim] = np.imag(input)

    rows, cols = weight.shape
    w_array = np.zeros((2*rows, 2*cols))
    w_array[0:rows, 0:cols] = np.real(weight)
    w_array[rows:, cols:] = np.imag(weight)

    whole_out = np.dot(in_whole, w_array)
    out_real = whole_out[0:no_batch, 0:cols] - whole_out[no_batch:, cols:]
    out_imag = whole_out[no_batch:, 0:cols] + whole_out[0:no_batch, cols:]

    return in_whole, w_array


def vmm_out_to_complex(input, weight, whole_out):
    no_batch, in_dim = input.shape
    rows, cols = weight.shape
    out_real = whole_out[0:no_batch, 0:cols] - whole_out[no_batch:, cols:]
    out_imag = whole_out[no_batch:, 0:cols] + whole_out[0:no_batch, cols:]
    ideal_out = np.dot(input,weight)
    vmm_out = out_real + out_imag * 1j
    return ideal_out, vmm_out


if __name__ == '__main__':
    # Read the data from the VMM post-processed data file.
    # The data is in the form of a pandas DataFrame.
    # data = pd.read_csv('data/VMM_post_processed_data.csv')
    # data = data.to_numpy()

    dft_mtx = DFT_mtx_create(16,16)
    plot_array(dft_mtx)
    # Extract the data fro
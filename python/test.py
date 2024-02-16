import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from VMM_sim import *
from Utility import *


def qtz_VMM(qtz_in, qtz_mtx):
    qtz_out = np.matmul(qtz_in, qtz_mtx)
    # Quantize the output matrix (Nxm) to the range [0, 255]
    min_out = np.min(qtz_out)
    max_out = np.max(qtz_out)
    qtz_out = np.round((qtz_out - min_out) / (max_out - min_out) * 255)
    return qtz_out


if __name__ == "__main__":
    qtz_in = np.load('/data/chaoyi_he/Crossbar_Sim/data/test_data_0216/int8_in_0216.npy')
    qtz_mtx = np.load('/data/chaoyi_he/Crossbar_Sim/data/test_data_0216/int8_weight_0216 (1).npy')
    npu_qtz_output = np.load('/data/chaoyi_he/Crossbar_Sim/data/test_data_0216/vmm_out_xb_0216.npy')
    
    ideal_qtz_output = qtz_VMM(qtz_in, qtz_mtx)
    
    # Plot the scatter plot of the ideal quantized output and the NPU quantized output separately
    fig = plt.figure()
    # Plot the scatter plot of ideal quantized output for each column
    ax = fig.add_subplot(2, 1, 1)
    columns = np.stack([np.arange(ideal_qtz_output.shape[1])] * ideal_qtz_output.shape[0], axis=0).flatten()
    ax.scatter(columns, ideal_qtz_output.flatten(), s=1)
    ax.set_title('Ideal quantized output')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Quantized output')
    ax.set_ylim([0, 255])
    # Plot the scatter plot of NPU quantized output for each column
    ax = fig.add_subplot(2, 1, 2)
    columns = np.stack([np.arange(ideal_qtz_output.shape[1])] * ideal_qtz_output.shape[0], axis=0).flatten()
    ax.scatter(columns, npu_qtz_output.flatten(), s=1)
    ax.set_title('NPU quantized output')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Quantized output')
    ax.set_ylim([0, 255])
    fig.savefig('./results/quantized_output_scatter_plot.png')
    
    # Plot the distribution of the quantized input
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(qtz_in.flatten(), bins=256, range=(0, 255))
    ax.set_title('Quantized input distribution')
    ax.set_xlabel('Quantized input')
    ax.set_ylabel('Frequency')
    fig.savefig('./results/quantized_input_distribution.png')
    
    # Plot the scatter plot of the ideal quantized output vs the NPU quantized output for column 10 for every 100 samples
    fig = plt.figure()
    for i in range(1):
        ax = fig.add_subplot(1, 1, i + 1)
        ax.scatter(ideal_qtz_output[i * 100:(i + 1) * 100, 10], npu_qtz_output[i * 100:(i + 1) * 100, 10], s=1)
        # ax.set_title('Sample ' + str(i))
        # ax.set_xlabel('Ideal quantized output')
        # ax.set_ylabel('NPU quantized output')
        # ax.set_xlim([0, 255])
        ax.set_ylim([0, 255])
    fig.savefig('./results/quantized_output_scatter_plot_10.png')
    
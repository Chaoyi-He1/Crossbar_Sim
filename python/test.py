import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from VMM_sim import *
from Utility import *


if __name__ == "__main__":
    ideal_qtz_output = np.load('./data/vmm_out_xb_in_dft_sim.npy')
    npu_qtz_output = np.load('./data/vmm_out_xb_in_dft.npy')
    
    # Plot the scatter plot of the ideal quantized output and the NPU quantized output separately
    fig = plt.figure()
    # Plot the scatter plot of ideal quantized output for each column
    ax = fig.add_subplot(2, 1, 1)
    columns = np.stack([np.arange(ideal_qtz_output.shape[1])] * ideal_qtz_output.shape[0], axis=0).flatten()
    ax.scatter(range(1600), ideal_qtz_output[:, 0], s=1)
    ax.set_title('Ideal quantized output')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Quantized output')
    ax.set_ylim([0, 255])
    # Plot the scatter plot of NPU quantized output for each column
    ax = fig.add_subplot(2, 1, 2)
    columns = np.stack([np.arange(ideal_qtz_output.shape[1])] * ideal_qtz_output.shape[0], axis=0).flatten()
    ax.scatter(range(1600), npu_qtz_output[:, 0], s=1)
    ax.set_title('NPU quantized output')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Quantized output')
    ax.set_ylim([0, 255])
    fig.savefig('./results/quantized_output_scatter_plot.png')
    # Plot the scatter plot of the ideal quantized output vs the NPU quantized output for column 10 for every 100 samples
    fig = plt.figure()
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.scatter(ideal_qtz_output[i * 100:(i + 1) * 100, 10], npu_qtz_output[i * 100:(i + 1) * 100, 10], s=1)
        ax.set_title('Sample ' + str(i))
        ax.set_xlabel('Ideal quantized output')
        ax.set_ylabel('NPU quantized output')
        # ax.set_xlim([0, 255])
        ax.set_ylim([0, 255])
    fig.savefig('./results/quantized_output_scatter_plot_10.png')
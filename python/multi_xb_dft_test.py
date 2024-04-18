import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from VMM_sim import *
from Utility import *
from multi_xb_VMM import *


if __name__ == "__main__":
    folder = "./data/test_data_dft_04_18/"
    recover_output = 'xb_recover_output_3_input_3_sin_40_100_rep_first_10.npy'
    ideal_output = 'ideal_recover_output_3_input_3_sin_40_100_rep_first_10.npy'
    
    recover_output = np.load(os.path.join(folder, recover_output))
    ideal_output = np.load(os.path.join(folder, ideal_output))
    # sample 10 rows for every 100 rows
    recover_output_resample = []
    for i in range(10):
        random_index = np.random.randint(0, 100, 10) + i * 100
        recover_output_resample.append(recover_output[random_index, :])
    recover_output_resample = np.vstack(recover_output_resample)
    
    recover_output_avg = np.mean(recover_output_resample, axis=0)
    
    #plot the recover_output_avg and ideal_output
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(recover_output_avg[1:], label='Recover Output Avg')
    plt.title('Recover Output')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(ideal_output[200, 1:], label='Ideal Output')
    plt.title('Ideal Output')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(folder, 'recover_output_vs_ideal_output.png'))
    
    pass
    
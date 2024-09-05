import time
import os
import datetime
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
from pathlib import Path


if __name__ == '__main__':
    I_data = np.load('/data/chaoyi_he/Crossbar_Sim/data/iq_demodulate/psk16_iq_cali_i.npy')
    Q_data = np.load('/data/chaoyi_he/Crossbar_Sim/data/iq_demodulate/psk16_iq_cali_q.npy')
    
    # I_data = I_data[:I_data.shape[0] // 2, :] - I_data[I_data.shape[0] // 2:, :]
    # Q_data = Q_data[:Q_data.shape[0] // 2, :] - Q_data[Q_data.shape[0] // 2:, :]
    
    # do overlap and add for I and Q
    I_data_seq = np.zeros((I_data.shape[0] * I_data.shape[1] // 2))
    Q_data_seq = np.zeros((Q_data.shape[0] * Q_data.shape[1] // 2))
    
    I_data_seq[0:I_data.shape[1]] = I_data[0]
    Q_data_seq[0:Q_data.shape[1]] = Q_data[0]
    
    previous_I = I_data[0]
    previous_Q = Q_data[0]
    for i in range(1, I_data.shape[0]):
        I_data_seq[i * I_data.shape[1] // 2 : (i + 1) * I_data.shape[1] // 2] = previous_I[I_data.shape[1] // 2:] + I_data[i][0:I_data.shape[1] // 2]
        Q_data_seq[i * Q_data.shape[1] // 2 : (i + 1) * Q_data.shape[1] // 2] = previous_Q[Q_data.shape[1] // 2:] + Q_data[i][0:Q_data.shape[1] // 2]
        previous_I = I_data[i]
        previous_Q = Q_data[i]
    
    #save the data to csv
    np.savetxt('/data/chaoyi_he/Crossbar_Sim/data/iq_demodulate/qam16_iq_cali_i_seq.csv', I_data_seq, delimiter=',')
    np.savetxt('/data/chaoyi_he/Crossbar_Sim/data/iq_demodulate/qam16_iq_cali_q_seq.csv', Q_data_seq, delimiter=',')
    np.savetxt('/data/chaoyi_he/Crossbar_Sim/data/iq_demodulate/qam16_iq_cali_i.csv', I_data, delimiter=',')
    np.savetxt('/data/chaoyi_he/Crossbar_Sim/data/iq_demodulate/qam16_iq_cali_q.csv', Q_data, delimiter=',')
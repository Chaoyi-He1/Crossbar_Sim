import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from VMM_sim import *
from Utility import *

def calibrate_p0_p1(test_data, ideal_data):
    """
    Do linear fitting between test data and ideal data for the VMM, and return the linear fitting parameters as p0 and p1.
    Note we assume N = 1 (batch size = 1).
    """
    channels = test_data.shape[-1]

    # Do linear fit to get p0,p1 for each channel.
    p0 = np.zeros(channels)
    p1 = np.zeros(channels)

    for k in range(channels):
        x = test_data[..., k]
        y = ideal_data[..., k]

        # ransac = linear_model.RANSACRegressor()
        # ransac.fit(x.reshape(-1, 1),y.reshape(-1, 1))

        # p0[k] = ransac.estimator_.coef_[0][0]
        # p1[k] = ransac.estimator_.intercept_[0]

        p = np.polyfit(x.reshape(-1), y.reshape(-1), 1)
        p0[k] = p[0]
        p1[k] = p[1]

    return p0, p1

def calibrate_data(test_data, p0, p1):
    """
    Calibrate the test data with the linear fitting parameters p0 and p1.
    """
    p0 = p0[np.newaxis, :]
    p1 = p1[np.newaxis, :]
    calibrated_data = np.round(test_data * p0 + p1)
    return calibrated_data


if __name__ == '__main__':
    v_range = [20, 240]
    g_range = [30, 235]

    float_in, float_weight, float_out = read_files('../data/input_for_calib.csv',
                                                   '../data/matrix_for_calib.csv',
                                                   '../data/output_for_calib.csv')
    exp_out = np.load('../data/vmm_out_fir_tamu.npz')['main_data']
    # float_out = np.dot(float_in, float_weight)
    Quan_out, a, b, c, d, max_range, min_range = Quantize_VMM(float_in, float_weight, v_range, g_range)
    
    #calibrate the output
    start_index = np.random.randint(0, float_out.shape[0] - 1000)
    p0, p1 = calibrate_p0_p1(exp_out, Quan_out)  # [start_index:start_index + 1000, :]
    exp_out_calib = calibrate_data(exp_out, p0, p1)
    
    Deduct_out_exp = Deduct_VM(exp_out_calib, a, b, c, d, max_range, min_range, float_in, float_weight)
    Deduct_out_ideal = Deduct_VM(Quan_out, a, b, c, d, max_range, min_range, float_in, float_weight)
    # Diff_out, a, b, c, d, max_range, min_range = diff_VMM(float_in, float_weight, v_range, g_range)
    
    # save the deducted output to a .csv file
    # save_output(Deduct_out_exp, './data/Deduct_out.csv')
    print("average difference between the deducted output and the ideal output is: ", 
          np.mean(np.abs(Deduct_out_exp - Deduct_out_ideal)))
    
    # plot float_out and Deduct_out and Diff_out in a figure with three subplots
    # randomly pick 512 consecutive elements from the output vectors
    start_index = np.random.randint(0, len(float_out) - 256)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(float_out.flatten()[start_index : start_index + 256], label='float_out')
    plt.legend()
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(Deduct_out_ideal.flatten()[start_index : start_index + 256], label='Deduct_out_ideal')
    plt.legend()
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(Deduct_out_exp.flatten()[start_index : start_index + 256], label='Diff_out_exp')
    plt.legend()
    plt.grid()
    plt.show()
    # save the figure
    plt.savefig('./results/float_out_vs_Deduct_out_ideal_vs_Deduct_out_exp.png')
    
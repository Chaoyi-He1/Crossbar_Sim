import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

import vmm_sim_pn
from vmm_post_process import *

if __name__ == '__main__':
    # Load the float point data
    float_in = pd.read_csv('data_for_umass/qam16/inputs.csv',header=None).to_numpy()
    float_weight = pd.read_csv('data_for_umass/qam16/conductance_matrix.csv',header=None).to_numpy()

    float_out = np.matmul(float_in, float_weight)
    # Assign the voltage range and conductance range
    # Both 0-255 DAC and ADC code, it is not recommanded to use extreme values close to 0 or 255
    v_range = [10, 246]
    # v_bit = 6
    v_bit = 4
    v_no = 2
    g_range = [127, 246]
    g_bit = 6
    g_no = 1
    out_range=[10,246]
    # 1-10, NPU index should be the same used in environment setting and fine-tuning command
    npu_index = 2
    # Chip name should be the same used in fine-tuning command
    # chip_name = 'umass_chip'
    # app_name = 'qam16_2in_pn'
    # Run quantized VMM simulation and return quantized voltage and conductance values
    vmm_sim_fir = vmm_sim_pn.VmmSim(float_in, float_weight, v_range,v_bit,v_no,g_range,g_bit,g_no, out_range)

    (int_out_pn,output_divisor_pn, qtz_in_pn, qtz_weight_pn,
      in_scale_pn, in_shift_pn, w_scale_pn, w_shift_pn, float_in_pn, float_weight_pn)\
        = vmm_sim_fir.quantize_vmm_pn()
    
    # w_scale_pn = np.diag(w_scale_pn)
    # w_shift_pn = np.ones(float_weight_pn.shape) * w_shift_pn
    # w_test = np.matmul(float_weight_pn - np.matmul(w_shift_pn, np.linalg.inv(w_scale_pn)), w_scale_pn) + w_shift_pn
    
    restore_out_sim_pn = vmm_sim_fir.reverse_vmm_pn(int_out_pn, output_divisor_pn, float_in_pn,
                                                         in_scale_pn, in_shift_pn,w_scale_pn, w_shift_pn)
    float_out_pn = np.matmul(float_in_pn,float_weight_pn)
    # np.save('../in_weight/' + 'int_input_' + app_name + '_pn_new.npy', qtz_in_pn)
    # np.save('../in_weight/' + 'int_weight_' + app_name + '_pn.npy', qtz_weight_pn)

    row_to_compare = 5

    xb_raw_out_pn = np.load('output_new.npy')

    p0_pn, p1_pn = calibrate_p0_p1(xb_raw_out_pn, int_out_pn)
    xb_out_calib_pn = calibrate_data(xb_raw_out_pn, p0_pn, p1_pn)
    restore_out_xb_pn = vmm_sim_fir.reverse_vmm_pn(xb_out_calib_pn, output_divisor_pn, float_in_pn,
                                                         in_scale_pn, in_shift_pn,w_scale_pn, w_shift_pn)

    fig = plt.figure(figsize=(10,8))
    # plt.title('XB results the ' + str(row_to_compare)+' row')
    ax1 = fig.add_subplot(311)
    ax1.plot(float_out[row_to_compare, :],'r',label = 'Float-point output')
    plt.legend()
    plt.grid(True)

    ax11 = fig.add_subplot(312)
    ax11.plot(restore_out_sim_pn[row_to_compare, :],'g',label = 'Simulation output')
    plt.legend()
    plt.grid(True)

    ax2 = fig.add_subplot(313)
    ax2.plot(restore_out_xb_pn[row_to_compare, :],'g',label = 'Crossbar output')
    plt.legend()
    plt.grid(True)

    plt.show()
    # save the plot
    fig.savefig('qam16_pn.png')

    err_pn = np.mean(np.abs(restore_out_xb_pn.astype(np.int32) + (restore_out_sim_pn[0, 0].astype(np.int32) - restore_out_xb_pn[0, 0].astype(np.int32)) - restore_out_sim_pn.astype(np.int32)))
    print('The mean error between the simulation and crossbar results is: ', err_pn)
    print("bias of crossbar output is: ", restore_out_xb_pn[0, 0].astype(np.int32))
    compare_output(xb_raw_out_pn, int_out_pn)
    
    #Overlap and add
    out_seq = np.zeros((1, 64 * 1000))
    previous_out = restore_out_xb_pn[0, :]
    for i in range(1000):
        out_cur = restore_out_xb_pn[i + 1, :]
        out_seq[0, i * 64: (i + 1) * 64] = out_cur[:64] + previous_out[64:]
        previous_out = restore_out_xb_pn[i + 1, :]
    
    out_seq_sim = np.zeros((1, 64 * 1000))
    previous_out = restore_out_sim_pn[0, :]
    for i in range(1000):
        out_cur = restore_out_sim_pn[i + 1, :]
        out_seq_sim[0, i * 64: (i + 1) * 64] = out_cur[:64] + previous_out[64:]
        previous_out = restore_out_sim_pn[i + 1, :]

    #FFT for the out_seq and out_seq_sim, plot the spectrum in two subplots
    fs = 20e3  # Sampling frequency
    n = len(out_seq[0])
    freq = np.fft.fftfreq(n, d=1/fs)

    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(211)
    ax1.plot(np.abs(np.fft.fft(out_seq[0, :]))[100:n//2])
    ax1.set_title('Crossbar output')
    ax2 = fig.add_subplot(212)
    ax2.plot(np.abs(np.fft.fft(out_seq_sim[0, :]))[100:n//2])
    ax2.set_title('Simulation output')
    plt.show()
    # save the plot
    fig.savefig('qam16_pn_fft.png')
    
    print('job done!')

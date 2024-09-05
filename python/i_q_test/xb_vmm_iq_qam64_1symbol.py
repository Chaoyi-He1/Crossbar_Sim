import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.io import loadmat
import pandas as pd
from xb_rf_sim import vmm_sim_run as vmm_sim_all
from xb_rf_sim.vmm_post_process import *
from xb_rf_sim.complex_vmm import *
# sys.path.append('./xb_rf_sim')

if __name__ == '__main__':
    # Load the float point data
    iq_in_data = pd.read_csv('python/i_q_test/data/PSK16/inputs_noise.csv', header=None)
    weight_i = pd.read_csv('python/i_q_test/data/PSK16/conductance_matrix_I.csv', header=None)
    weight_q = pd.read_csv('python/i_q_test/data/PSK16/conductance_matrix_Q.csv', header=None)

    float_in = iq_in_data.values
    float_weight_i = weight_i.values
    float_weight_q = weight_q.values
    float_weight = np.concatenate((float_weight_i,float_weight_q),axis=1)

    float_out_i = np.matmul(float_in, float_weight_i)
    float_out_q = np.matmul(float_in, float_weight_q)
    # Assign the voltage range and conductance range
    # Both 0-255 DAC and ADC code, it is not recommanded to use extreme values close to 0 or 255
    v_range = [50, 246]
    v_bit = 4
    v_no = 2
    # g_range = [40, 170]
    g_range = [127, 246]
    g_bit = 6
    g_no = 1
    out_range=[10, 246]
    # 1-10, NPU index should be the same used in environment setting and fine-tuning command
    npu_index = 2
    # Chip name should be the same used in fine-tuning command
    chip_name = 'umass_chip'
    app_name = 'iq_qam64_1symbol_0815'
    # Run quantized VMM simulation and return quantized voltage and conductance values
    vmm_sim_i = vmm_sim_all.VmmSim(float_in, float_weight_i, v_range,v_bit,v_no,g_range,g_bit,g_no, out_range)
    (int_out_i, output_divisor_i, qtz_in_i, qtz_weight_i,
      in_scale_i, in_shift_i, w_scale_i, w_shift_i, float_in_i)\
        = vmm_sim_i.quantize_vmm_pn()
    restore_out_sim_i = vmm_sim_i.reverse_vmm_pn(int_out_i, output_divisor_i, float_in_i,
                                                         in_scale_i, in_shift_i,w_scale_i, w_shift_i)

    vmm_sim_q= vmm_sim_all.VmmSim(float_in, float_weight_q, v_range,v_bit,v_no,g_range,g_bit,g_no, out_range)
    (int_out_q, output_divisor_q, qtz_in_q, qtz_weight_q,
      in_scale_q, in_shift_q, w_scale_q, w_shift_q, float_in_q)\
        = vmm_sim_q.quantize_vmm_pn()
    restore_out_sim_q = vmm_sim_q.reverse_vmm_pn(int_out_q, output_divisor_q, float_in_q,
                                                         in_scale_q, in_shift_q, w_scale_q, w_shift_q)

    # np.save('../in_weight/' + 'int_input_' + app_name + '_i.npy', qtz_in_i)
    # np.save('../in_weight/' + 'int_weight_' + app_name + '_i.npy', qtz_weight_i)
    # np.save('../in_weight/' + 'int_input_' + app_name + '_q.npy', qtz_in_q)
    # np.save('../in_weight/' + 'int_weight_' + app_name + '_q.npy', qtz_weight_q)

    #
    # for i in range(qtz_in_pn.shape[0]):
    #     plt.scatter(range(0,qtz_in_pn.shape[1]),qtz_in_pn[i,:])
    #     plt.show()
    # row_to_compare = 1
    # #
    # xb_raw_out_i = np.zeros((5,int_out_i.shape[0],int_out_i.shape[1]))
    # xb_out_calib_i = np.zeros((5,int_out_i.shape[0],int_out_i.shape[1]))
    # for i in range(5):
    #     xb_raw_out_i[i] = np.load(
    #         '../results/npu_mat_mul_'+app_name+'_i_core' + str(i+1)+'/hw_test_e2e/output/output.npy')
    #     p0_i, p1_i = calibrate_p0_p1(xb_raw_out_i[i], int_out_i)
    #     xb_out_calib_i[i] = calibrate_data(xb_raw_out_i[i], p0_i, p1_i)
    # xb_out_calib_i_mean = np.mean(xb_out_calib_i, axis=0)
    # # restore_out_xb_pn = vmm_sim_iq.reverse_vmm_pn(xb_out_calib_pn, output_divisor_pn, float_in_pn,
    # #                                                      in_scale_pn, in_shift_pn,w_scale_pn, w_shift_pn)
    # # restore_output_xb_mean = np.mean(restore_out_xb_pn, axis=0)
    # xb_raw_out_q = np.zeros((5,int_out_q.shape[0],int_out_q.shape[1]))
    # xb_out_calib_q = np.zeros((5,int_out_q.shape[0],int_out_q.shape[1]))
    # for i in range(5):
    #     xb_raw_out_q[i] = np.load(
    #         '../results/npu_mat_mul_'+app_name+'_q_core' + str(i+1)+'/hw_test_e2e/output/output.npy')
    #     p0_q, p1_q = calibrate_p0_p1(xb_raw_out_q[i], int_out_q)
    #     xb_out_calib_q[i] = calibrate_data(xb_raw_out_q[i], p0_q, p1_q)
    #
    # xb_out_calib_q_mean = np.mean(xb_out_calib_q, axis=0)
    # fig = plt.figure(figsize=(10,10))
    # # plt.title('XB results the ' + str(row_to_compare)+' row')
    # ax1 = fig.add_subplot(411)
    # ax1.plot(float_out_i[row_to_compare, :],'r',label = 'Ideal output')
    # plt.legend()
    # plt.grid(True)
    #
    # ax2 = fig.add_subplot(412)
    # ax2.plot(restore_out_sim_i[row_to_compare, :],'g',label = 'Simulation output')
    # plt.legend()
    # plt.grid(True)
    #
    # ax3 = fig.add_subplot(413)
    # ax3.plot(xb_out_calib_i[0,row_to_compare,:],'b',label = 'Experimental output')
    # plt.legend()
    # plt.grid(True)
    # #
    # ax4 = fig.add_subplot(414)
    # ax4.plot(xb_out_calib_i_mean[row_to_compare,:],'b',label = 'Average PN output')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.show()
    # fig = plt.figure(figsize=(10,10))
    # # plt.title('XB results the ' + str(row_to_compare)+' row')
    # ax1 = fig.add_subplot(411)
    # ax1.plot(float_out_q[row_to_compare, :],'r',label = 'Ideal output')
    # plt.legend()
    # plt.grid(True)
    #
    # ax2 = fig.add_subplot(412)
    # ax2.plot(restore_out_sim_q[row_to_compare, :],'g',label = 'Simulation output')
    # plt.legend()
    # plt.grid(True)
    #
    # ax3 = fig.add_subplot(413)
    # ax3.plot(xb_out_calib_q[0,row_to_compare,:],'b',label = 'Experimental output')
    # plt.legend()
    # plt.grid(True)
    # #
    # ax4 = fig.add_subplot(414)
    # ax4.plot(xb_out_calib_q_mean[row_to_compare,:],'r',label = 'Average PN output')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.show()
    # # compare_output(xb_raw_out_i[0], int_out_i)
    # # compare_output(xb_raw_out_q[0], int_out_q)
    #
    # np.save('../results/qam16_iq_cali_i_average.npy', xb_out_calib_i_mean)
    # np.save('../results/qam16_iq_cali_q_average.npy', xb_out_calib_q_mean)
    # np.save('../results/qam16_iq_sim_i.npy', int_out_i)
    # np.save('../results/qam16_iq_sim_q.npy', int_out_q)
    #
    # err_pn = np.mean(np.abs(restore_output_xb_mean.astype(np.int32) - np.mean(restore_out_sim_pn).astype(np.int32)))
    # # compare_output(xb_raw_out_pn, int_out_pn)
    #
    # print('job done!')

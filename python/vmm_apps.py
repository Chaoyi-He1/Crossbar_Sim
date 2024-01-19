import numpy as np
import sys
# sys.path.append('./Crossbar_Sim/python')

from Crossbar_Sim.python.VMM_sim import Quantize_VMM
from Crossbar_Sim.python.VMM_sim import Deduct_VM
from Crossbar_Sim.python.visualize import plot_array
from Crossbar_Sim.python.VMM_post_process import *

# from flow_file_generate import file_generate

if __name__ == '__main__':

    # Load the float point data
    float_weight = np.load('../data/float_weight_example.npy')
    float_in = np.load('../data/float_in_example.npy')
    float_out = np.dot(float_in, float_weight)

    # Assign the voltage range and conductance range
    # Both 0-255 DAC and ADC code, it is not recommanded to use extreme values close to 0 or 255
    v_range = [20, 240]
    g_range = [40, 235]
    # 1-10, NPU index should be the same used in environment setting and fine-tuning command
    npu_index = 1
    # Chip name should be the same used in fine-tuning command
    chip_name = 'umass_chip'
    # The application name to run
    app_name = 'example'

    # Run quantized VMM simulation and return quantized voltage and conductance values
    Quan_out, a, b, c, d, max_range, min_range, qtz_v, qtz_g = Quantize_VMM(float_in, float_weight, v_range, g_range)

    # Generate json and yaml files for TetraMem SDK
    # Comment out when run post processing
    # xb_array = {app_name : qtz_g}
    # file_generate(xb_array, qtz_v.T, chip_name, npu_index, app_name)

    # After run the experimental
    xb_g = np.loadtxt('../results/read_out_' + app_name + '.txt')
    with np.load('../results/vmm_out_' + app_name + '.npz') as xb_data:
         xb_vmm_out = xb_data['main_data']

    # Number of data used for calibration
    no_cali = 1000
    # Calibrate the outputs
    p0, p1 = calibrate_p0_p1(xb_vmm_out[0:no_cali], Quan_out[0:no_cali])  # [start_index:start_index + 1000, :]
    xb_out_calib = calibrate_data(xb_vmm_out, p0, p1)

    Deduct_out = Deduct_VM(Quan_out, a, b, c, d, max_range, min_range, float_in, float_weight)
    Deduct_out_xb = Deduct_VM(xb_out_calib, a, b, c, d, max_range, min_range, float_in, float_weight)

    # save the deducted output to a .csv file
    # save_output(Deduct_out_xb, './results/Deduct_out_xb.csv')
    print("average difference between the deducted output and the ideal output is: ",
          np.mean(np.abs(Deduct_out_xb - Deduct_out)))
    # Plot target and experimental conductance
    plot_array(qtz_g)
    plot_array(xb_g)
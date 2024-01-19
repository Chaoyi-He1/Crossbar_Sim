import numpy as np
import sys
from VMM_sim import Quantize_VMM
from VMM_sim import Deduct_VM
from visualize import plot_array
from VMM_post_process import *


if __name__ == '__main__':
    float_weight_all = np.loadtxt('/data/chaoyi_he/Crossbar_Sim/data/matrix_for_calib_sin.csv', delimiter=',')
    float_in_all = np.loadtxt('/data/chaoyi_he/Crossbar_Sim/data/input_for_calib_sin.csv', delimiter=',')
    float_in_1 = np.zeros((1000, 128))
    float_in_2 = np.zeros((1000, 128))
    float_weight_1 = np.zeros((128, 256))
    float_weight_2 = np.zeros((128, 256))
    split_weight = np.split(float_weight_all,8,axis=0)
    split_in = np.split(float_in_all,8,axis=0)

    for i in range(8):
        if i<4:
            float_weight_1[i * 32:(i + 1) * 32, i * 64:(i + 1) * 64] = split_weight[i]
            float_in_1[:, i * 32:(i + 1) * 32] = split_in[i]
        else:
            float_weight_2[(i-4) * 32:(i - 3) * 32, (i - 4) * 64:(i - 3) * 64] = split_weight[i]
            float_in_2[:, (i - 4) * 32:(i -3) * 32] = split_in[i]

    float_out_1 = np.dot(float_in_1, float_weight_1)
    float_out_2 = np.dot(float_in_2, float_weight_2)

    split_out_1 = np.split(float_out_1,4,axis=1)
    split_out_2 = np.split(float_out_2, 4, axis=1)
    out_part_1 = np.concatenate(split_out_1,axis=0)
    out_part_2 = np.concatenate(split_out_2,axis=0)

    float_out_all = np.concatenate([out_part_1,out_part_2],axis=0)

    ideal_out = np.loadtxt('/data/chaoyi_he/Crossbar_Sim/data/output_for_calib_sin.csv', delimiter=',')

    # ideal_float = (float_out_all == ideal_out)
    print('diff average: ', np.mean(np.abs(float_out_all - ideal_out)))

import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt


'''
The function takes an input voltage vector and conductance matrix,
together with voltage and conductance quantization ranges, and
returns the VMM output vector requantized to the specified ranges [0, 255]
'''
def Quantize_VMM(voltages, conductances, v_range, g_range):
    '''
    Parameters:
        voltages: input voltage vectors (Nxl), where N is the number of vectors and l is the length of each vector
        conductances: conductance matrix (lxm), where l is the length of each vector and m is the output vector length
        v_range: voltage quantization range (2x1), where the first element is the minimum and the second element is the maximum
                 The voltages will be quantized to integers in the range [v_range[0], v_range[1]]
        g_range: conductance quantization range (2x1), where the first element is the minimum and the second element is the maximum
                 The conductances will be quantized to integers in the range [g_range[0], g_range[1]]
    Returns:
        out: output vector (Nxm), where N is the number of vectors and m is the output vector length
             the output is originally in the range [v_range[0] * g_range[0], v_range[1] * g_range[1]],
             then it is quantized to the range [0, 255]
        a: the scaling factor used to scale the voltages to the range [v_range[0], v_range[1]]
        b: the bias used to scale the voltages to the range [v_range[0], v_range[1]]
        c: the scaling factor used to scale the conductances to the range [g_range[0], g_range[1]]
        d: the bias used to scale the conductances to the range [g_range[0], g_range[1]] 
        max_out: the maximum value of the output matrix
        min_out: the minimum value of the output matrix
    The quantization will quantize the voltages and conductances to integers in the range "v_range" and "g_range" respectively
    '''
    min_v = np.min(voltages)
    max_v = np.max(voltages)
    min_g = np.min(conductances)
    max_g = np.max(conductances)
    
    # Quantize the voltages and conductances to integers in the range "v_range" and "g_range" respectively
    qtz_voltages = np.round((voltages - min_v) / (max_v - min_v) * (v_range[1] - v_range[0]) + v_range[0])
    qtz_conductances = np.round((conductances - min_g) / (max_g - min_g) * (g_range[1] - g_range[0]) + g_range[0])
    
    a = (v_range[1] - v_range[0]) / (max_v - min_v)
    b = v_range[0] - a * min_v
    c = (g_range[1] - g_range[0]) / (max_g - min_g)
    d = g_range[0] - c * min_g
    
    # Compute the output vector
    out = np.matmul(qtz_voltages, qtz_conductances)
    
    # Quantize the output vector to the range [0, 255]
    min_out = np.min(out)
    max_out = np.max(out)
    qtz_out = np.round((out - min_out) / (max_out - min_out) * 255)
    
    return qtz_out, a, b, c, d, max_out, min_out


'''
The function takes the quantized output vectors "out" in the range [0, 255], which is the result of (aV+b)*(cM+d),
the scaling factors and biases, (a, b, c, d), 
and v_range and g_range, which are the voltage and conductance quantization ranges respectively.

(aV+b) is in the range [v_range[0], v_range[1]] and (cM+d) is in the range [g_range[0], g_range[1]],
and the original input voltage vector and conductance matrix, V and M

The function is to rescale the output vectors from the range [0, 255] to the range [v_range[0] * g_range[0], v_range[1] * g_range[1]]
and then subtract the terms "adV", "bcM" and "bd" from the output vectors.
The function then quantizes the output vector acVM, deducted the terms "adV", "bcM" and "bd" back to the range [0, 255]
'''
def Deduct_VM(out, a, b, c, d, max_range, min_range, V, M):
    
    '''
    Parameters:
        out: output vector (Nxm), where N is the number of vectors and m is the output vector length
             the output is originally in the range [v_range[0] * g_range[0], v_range[1] * g_range[1]],
             then it is quantized to the range [0, 255]
        a: the scaling factor used to scale the voltages to the range [v_range[0], v_range[1]]
        b: the bias used to scale the voltages to the range [v_range[0], v_range[1]]
        c: the scaling factor used to scale the conductances to the range [g_range[0], g_range[1]]
        d: the bias used to scale the conductances to the range [g_range[0], g_range[1]] 
        max_range: the maximum value of the output matrix before quantization to the range [0, 255]
        min_range: the minimum value of the output matrix before quantization to the range [0, 255]
        V: input voltage vectors (Nxl) as a numpy array, where N is the number of vectors and l is the length of each vector
        M: conductance matrix (lxm) as a numpy array, where l is the length of each vector and m is the output vector length
    Returns:
        out: output vector (Nxm), where N is the number of vectors and m is the output vector length
             the output is originally in the range [0, 255] with integer values, the output includes all the terms
             "acVM + adV + bcM + bd",
             then it is rescaled to the range [v_range[0] * g_range[0], v_range[1] * g_range[1]] in order to deduct
             the terms "adV", "bcM" and "bd", and then quantized back to the range [0, 255]
    '''
    # Rescale the output vector to the range [v_range[0] * g_range[0], v_range[1] * g_range[1]]
    min_out = np.min(out)
    max_out = np.max(out)
    rescaled_out = (out - min_out) / (max_out - min_out) * (max_range - min_range) + min_range
    # diff = rescaled_out - out
    
    # Deduct the terms "adV", "bcM" and "bd" from the output vector
    adV = a * np.matmul(V, d * np.ones((M.shape[0], M.shape[1])))
    bcM = c * np.matmul(b * np.ones((V.shape[0], V.shape[1])), M)
    bd = np.matmul(b * np.ones((V.shape[0], V.shape[1])), d * np.ones((M.shape[0], M.shape[1])))
    out = rescaled_out - adV - bcM - bd
    
    # Quantize the output vector to the range [0, 255]
    min_out = np.min(out)
    max_out = np.max(out)
    out = np.round((out - min_out) / (max_out - min_out) * 255)
    
    return out


'''
This function is to real the input voltage vectors, conductance matrix and theoretical output vectors 
from the input .csv file
'''
def read_files(input_V_file, conductance_file, output_file):
    '''
    Parameters:
        input_V_file: the input voltage vectors file name
        conductance_file: the conductance matrix file name
        output_file: the theoretical output vectors file name
    Returns:
        V: input voltage vectors (Nxl) as a numpy array, where N is the number of vectors and l is the length of each vector
        M: conductance matrix (lxm) as a numpy array, where l is the length of each vector and m is the output vector length
        out: theoretical output vectors (Nxm) as a numpy array, where N is the number of vectors and m is the output vector length
    '''
    # Read the input voltage vectors from the input .csv file
    Voltage = pd.read_csv(input_V_file, header=None).to_numpy()
    conductance = pd.read_csv(conductance_file, header=None).to_numpy()
    ideal_out = pd.read_csv(output_file, header=None).to_numpy()
    
    return Voltage, conductance, ideal_out


'''
This function saves the output vector to the output .csv file
'''
def save_output(out, output_file):
    '''
    Parameters:
        out: output vector (Nxm), where N is the number of vectors and m is the output vector length
        output_file: the output .csv file name
    '''
    # Save the output vector to the output .csv file
    pd.DataFrame(out).to_csv(output_file, header=None, index=None)


if __name__ == '__main__':
    input_dim = 20
    no_batch = 2
    output_dim = 30
    v_range = [10, 245]
    g_range = [15, 240]

    float_weight = np.random.randn(input_dim, output_dim)
    float_in = np.random.randn(no_batch, input_dim)
    float_out = np.dot(float_in, float_weight)
    Quan_out, a, b, c, d, max_range, min_range = Quantize_VMM(float_in, float_weight, v_range, g_range)
    Deduct_out = Deduct_VM(Quan_out, a, b, c, d, max_range, min_range, float_in, float_weight)
    
    # plot float_out and Deduct_out in a figure with two subplots
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(float_out.flatten(),label='float_out')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(Deduct_out.flatten(),label='Deduct_out')
    plt.legend()
    plt.show()
    # save the figure
    plt.savefig('float_out_Deduct_out.png')
    
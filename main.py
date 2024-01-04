import numpy as np
import pandas as pd
import os
import math


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
    The quantization will quantize the voltages and conductances to integers in the range "v_range" and "g_range" respectively
    '''
    min_v = np.min(voltages)
    max_v = np.max(voltages)
    min_g = np.min(conductances)
    max_g = np.max(conductances)
    
    # Quantize the voltages and conductances to integers in the range "v_range" and "g_range" respectively
    qtz_voltages = np.round((voltages - min_v) / (max_v - min_v) * (v_range[1] - v_range[0]) + v_range[0])
    qtz_conductances = np.round((conductances - min_g) / (max_g - min_g) * (g_range[1] - g_range[0]) + g_range[0])
    
    a = (max_v - min_v) / (v_range[1] - v_range[0])
    b = v_range[0] - a * min_v
    c = (max_g - min_g) / (g_range[1] - g_range[0])
    d = g_range[0] - c * min_g
    
    # Compute the output vector
    out = np.matmul(qtz_voltages, qtz_conductances)
    
    # Quantize the output vector to the range [0, 255]
    min_out = np.min(out)
    max_out = np.max(out)
    out = np.round((out - min_out) / (max_out - min_out) * 255)
    
    return out, a, b, c, d


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
def Deduct_VM(out, a, b, c, d, v_range, g_range):
    '''
    Parameters:
        out: output vector (Nxm), where N is the number of vectors and m is the output vector length
             the output is originally in the range [v_range[0] * g_range[0], v_range[1] * g_range[1]],
             then it is quantized to the range [0, 255]
        a: the scaling factor used to scale the voltages to the range [v_range[0], v_range[1]]
        b: the bias used to scale the voltages to the range [v_range[0], v_range[1]]
        c: the scaling factor used to scale the conductances to the range [g_range[0], g_range[1]]
        d: the bias used to scale the conductances to the range [g_range[0], g_range[1]] 
        v_range: voltage quantization range (2x1), where the first element is the minimum and the second element is the maximum
                 The voltages will be quantized to integers in the range [v_range[0], v_range[1]]
        g_range: conductance quantization range (2x1), where the first element is the minimum and the second element is the maximum
                 The conductances will be quantized to integers in the range [g_range[0], g_range[1]]
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
    rescaled_out = (out - min_out) / (max_out - min_out) * (v_range[1] * g_range[1] - v_range[0] * g_range[0]) + v_range[0] * g_range[0]
    
    # Deduct the terms "adV", "bcM" and "bd" from the output vector
    adV = a * d * v_range[0]
    bcM = b * c * g_range[0]
    bd = b * d
    out = rescaled_out - adV - bcM - bd
    
    # Quantize the output vector to the range [0, 255]
    min_out = np.min(out)
    max_out = np.max(out)
    out = np.round((out - min_out) / (max_out - min_out) * 255)
    
    return out
    
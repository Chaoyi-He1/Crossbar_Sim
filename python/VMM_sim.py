import numpy as np

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
This function is to simulate the VMM with differential pairs
The matrix will have two diagonal parts, the first part is the normal matrix to be quantized in the range [g_range[0], g_range[1]]
The second part is the median value of the quantization range, which is the value to be subtracted from the output vector
if the conductance range has odd number of values, the median value is the middle value of the range
if the conductance range has even number of values, the median value should be calculated as the average of the two middle values
'''


def diff_VMM(voltages, conductances, v_range, g_range):
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

    median_g = (g_range[0] + g_range[1]) / 2

    a = (v_range[1] - v_range[0]) / (max_v - min_v)
    b = v_range[0] - a * min_v
    c = (g_range[1] - g_range[0]) / (max_g - min_g)
    d = g_range[0] - c * min_g

    # Compute the output vector
    out = np.matmul(qtz_voltages, qtz_conductances)
    median_out = np.matmul(qtz_voltages, median_g * np.ones((qtz_conductances.shape[0], qtz_conductances.shape[1])))

    min_out = np.min(out)
    max_out = np.max(out)

    out = out - c * np.matmul(b * np.ones((qtz_voltages.shape[0], qtz_voltages.shape[1])), conductances)

    # Quantize the output vector to the range [0, 255]
    qtz_out = np.round((out - min_out) / (max_out - min_out) * 255)
    qtz_median_out = np.round((median_out - min_out) / (max_out - min_out) * 255)
    qtz_out = qtz_out - qtz_median_out
    qtz_out = np.round((qtz_out - np.min(qtz_out)) / (np.max(qtz_out) - np.min(qtz_out)) * 255)

    return qtz_out, a, b, c, d, max_out, min_out

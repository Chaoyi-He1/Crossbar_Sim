import numpy as np
import sys
from visualize import plot_array


def Quantize_input(input_vectors, v_range, num_intervals=16):
    '''
    Parameters:
        voltages: input voltage vectors (Nxl), where N is the number of vectors and l is the length of each vector
        v_range: voltage quantization range (2x1), where the first element is the minimum and the second element is the maximum
                 The voltages will be quantized to integers in the range [v_range[0], v_range[1]]
    Returns:
        qtz_voltages: quantized input voltage vectors (Nxl), where N is the number of vectors and l is the length of each vector
        qtz_indices: the indices of the quantized input voltage vectors (Nxl), where N is the number of vectors and l is the length of each vector
        interval_widths: the width of each interval (N, 1)
    '''
    # Determine codes and prepare for broadcasting
    codes = np.linspace(v_range[0], v_range[1], num_intervals, endpoint=False, dtype=int)
    
    # Compute min and max for each vector
    min_vals = np.min(input_vectors)
    max_vals = np.max(input_vectors)
    
    # Calculate interval widths for each vector
    interval_widths = (max_vals - min_vals) / num_intervals
    
    # Normalize vectors to range [0, 1] within their respective min-max range
    normalized_vectors = (input_vectors - min_vals) / (max_vals - min_vals)
    
    # Scale normalized vectors to code indices and clip to valid range
    scaled_indices = np.floor(normalized_vectors * num_intervals).astype(int)
    scaled_indices = np.clip(scaled_indices, 0, num_intervals - 1)
    
    # Map indices to quantized voltage values
    quantized_lvs = codes[scaled_indices]
    
    return quantized_lvs, scaled_indices, interval_widths


'''
This function is to implement the VMM with multiple crossbars
The function will separate the input with multiple levels first.
Then it will implement the VMM with each level separately.
The output of each level will be added together to get the final output

The first crossbar will quantize the input to the range [v_range[0], v_range[1]] ([0, 255] int) with 16 levels: 
    [min(input), min(input)+(max(input)-min(input))/16, min(input)+2*(max(input)-min(input))/16, ..., max(input)] to:
    [v_range[0], int(v_range[0]+(v_range[1]-v_range[0])/16), int(v_range[0]+2*(v_range[1]-v_range[0])/16), ..., v_range[1]]  
Then, the second crossbar will quantize the input with further precision to the range [v_range[0], v_range[1]] ([0, 255] int) with 16 levels:
    for each interval in the first crossbar, it will be quantized to 16 levels further:
    [min(input)+n*(max(input)-min(input))/16,  min(input)+n*(max(input)-min(input))/16+(max(input)-min(input))/16/16, ...] to:
    [v_range[0], int(v_range[0]+(v_range[1]-v_range[0])/16), int(v_range[0]+2*(v_range[1]-v_range[0])/16), ..., v_range[1]]
Then, the third crossbar will quantize the input with further precision to the range [v_range[0], v_range[1]] ([0, 255] int) with 16 levels:
    for each interval in the second crossbar, it will be quantized to 16 levels further:
    [min(input)+n*(max(input)-min(input))/16+m*(max(input)-min(input))/16/16,  min(input)+n*(max(input)-min(input))/16+m*(max(input)-min(input))/16+(max(input)-min(input))/16/16/16, ...] to:
    [v_range[0], int(v_range[0]+(v_range[1]-v_range[0])/16), int(v_range[0]+2*(v_range[1]-v_range[0])/16), ..., v_range[1]]
The the VMM will be implemented with the quantized input and conductance matrix
The output of each level will be added together to get the final output:
    out = (n*(max(input)-min(input))/16 + (m-128)*(max(input)-min(input))/16/16 + (l-128)*(max(input)-min(input))/16/16/16) * conductance_matrix

'''
def VMM_with_multi_XB(voltages, conductances, v_range, g_range, num_xbars, num_steps=16):
    '''
    Parameters:
        voltages: input voltage vectors (N x l), where N is the number of vectors and l is the length of each vector
        conductances: conductance matrix (lxm), where l is the length of each vector and m is the output vector length
        v_range: voltage quantization range (2x1), where the first element is the minimum and the second element is the maximum
                 The voltages will be quantized to integers in the range [v_range[0], v_range[1]]
        g_range: conductance quantization range (2x1), where the first element is the minimum and the second element is the maximum
                 The conductances will be quantized to integers in the range [g_range[0], g_range[1]]
        num_xbars: the number of crossbars to quantize the input
    Returns:
        out: output vector (num_xbars x N x m), where N is the number of vectors and m is the output vector length
             the output is originally in the range [v_range[0] * g_range[0], v_range[1] * g_range[1]],
             then it is quantized to the range [0, 255]
        a: the size is (num_xbars, 1), the scaling factor used to scale the voltages to the range [v_range[0], v_range[1]]
        b: the size is (num_xbars, 1), the bias used to scale the voltages to the range [v_range[0], v_range[1]]
        c: the size is (1, m), the scaling factor used to scale the conductances to the range [g_range[0], g_range[1]] column-wise
        d: the size is (1, m), the bias used to scale the conductances to the range [g_range[0], g_range[1]] column-wise
        max_out: the maximum value of each output matrix, the size is (num_xbars, )
        min_out: the minimum value of each output matrix, the size is (num_xbars, )
    '''
    
    # Quantize the conductances to integers in the range "g_range" column-wise
    min_g = np.min(conductances, axis=0, keepdims=True)
    max_g = np.max(conductances, axis=0, keepdims=True)
    
    codes_g = np.arange(g_range[0], g_range[1] + 1, dtype=int)
    qtz_conductances = codes_g[
        np.clip(
            np.floor((conductances - min_g) / (max_g - min_g) * len(codes_g)).astype(int), 
            0, len(codes_g) - 1)]
    
    # Compute c, d
    c = (g_range[1] - g_range[0]) / (max_g - min_g)
    d = g_range[0] - c * min_g
    
    # Quantize the voltages to integers in the range "v_range" by num_xbars crossbars
    qtz_input = np.zeros((num_xbars, voltages.shape[0], conductances.shape[1]))
    qtz_input_index = np.zeros((num_xbars, voltages.shape[0], voltages.shape[1]))
    input_interval_widths = np.zeros((num_xbars, 1))
    
    qtz_output = np.zeros((num_xbars, voltages.shape[0], conductances.shape[1]))
    max_out = np.zeros(num_xbars, 1)
    min_out = np.zeros(num_xbars, 1)
    a, b = np.zeros((num_xbars, 1)), np.zeros((num_xbars, 1))
    
    for i in range(num_xbars):
        # Compute a, b
        max_v = np.max(voltages)
        min_v = np.min(voltages)
        a[i, :] = (v_range[1] - v_range[0]) / (max_v - min_v)
        b[i, :] = v_range[0] - a[i, :] * min_v
        
        (qtz_input[i, :, :], 
         qtz_input_index[i, :, :], 
         input_interval_widths[i, :]) = Quantize_input(voltages, v_range, num_steps)
        
        voltages = voltages - input_interval_widths[i, :] * qtz_input_index[i, :, :]
        
        # Compute the output vector and quantize it to the range [0, 255]
        out = np.matmul(qtz_input[i, :, :], qtz_conductances)
        min_out[i, :], max_out[i, :] = np.min(out), np.max(out)
        
        codes_out = np.arange(0, 256, dtype=int)
        qtz_output[i, :, :] = codes_out[
            np.clip(
                np.floor((out - min_out[i, :]) / (max_out[i, :] - min_out[i, :]) * 255).astype(int), 
                0, 255)]
    
    return qtz_output, a, b, c, d, max_out, min_out, qtz_input, qtz_input_index, input_interval_widths, qtz_conductances



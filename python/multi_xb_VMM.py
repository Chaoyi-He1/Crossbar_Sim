import numpy as np
import sys
from visualize import plot_array
from VMM_sim import Quantize_VMM, Deduct_VM
import matplotlib.pyplot as plt


def Quantize_input(input_vectors, v_range, interval_width, num_intervals=16):
    '''
    Parameters:
        voltages: input voltage vectors (Nxl), where N is the number of vectors and l is the length of each vector
        v_range: voltage quantization range (2x1), where the first element is the minimum and the second element is the maximum
                 The voltages will be quantized to integers in the range [v_range[0], v_range[1]]
    Returns:
        quantized_lvs: quantized input voltage vectors (Nxl, [v_range[0], v_range[1]]), where N is the number of vectors and l is the length of each vector
        scaled_indices: the indices of the quantized input voltage vectors (Nxl)
    '''
    # Determine codes and prepare for broadcasting
    codes = np.linspace(v_range[0], v_range[1], num_intervals, endpoint=True, dtype=int)
    
    # Compute float voltage range
    float_v_range = interval_width * num_intervals
    # Compute a and b
    a = (v_range[1] - v_range[0]) / float_v_range
    b = v_range[0]
    
    # Normalize vectors to range [0, 1] within their respective min-max range
    normalized_vectors = input_vectors / float_v_range
    
    # Scale normalized vectors to code indices and clip to valid range
    scaled_indices = np.floor(normalized_vectors * num_intervals).astype(int)
    scaled_indices = np.clip(scaled_indices, 0, num_intervals - 1)
    
    # Map indices to quantized voltage values
    quantized_lvs = codes[scaled_indices]
    
    return quantized_lvs, scaled_indices, a, b


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
    total_min_v = np.min(voltages)
    
    # Quantize the conductances to integers in the range "g_range" column-wise
    min_g = np.min(conductances, axis=0, keepdims=True)
    max_g = np.max(conductances, axis=0, keepdims=True)
    
    codes_g = np.arange(g_range[0], g_range[1] + 1, dtype=int)
    qtz_conductances = codes_g[
        np.clip(
            np.floor((conductances - min_g) / (max_g - min_g) * (len(codes_g) - 1)).astype(int), 
            0, len(codes_g) - 1)]
    
    # Compute c, d
    c = (g_range[1] - g_range[0]) / (max_g - min_g)
    d = g_range[0] - c * min_g
    
    # Quantize the voltages to integers in the range "v_range" by num_xbars crossbars
    qtz_input = np.zeros((num_xbars, voltages.shape[0], voltages.shape[1]))
    qtz_input_index = np.zeros((num_xbars, voltages.shape[0], voltages.shape[1]))
    input_interval_widths = np.zeros((num_xbars, 1))
    
    qtz_output = np.zeros((num_xbars + 1, voltages.shape[0], conductances.shape[1]))
    qtz_output_deduct = np.zeros((num_xbars + 1, voltages.shape[0], conductances.shape[1]))
    max_out = np.zeros((num_xbars + 1, 1))
    min_out = np.zeros((num_xbars + 1, 1))
    a, b = np.zeros((num_xbars + 1, 1)), np.zeros((num_xbars + 1, 1))
    rescale_factor, rescale_bias = np.zeros((num_xbars + 1, 1)), np.zeros((num_xbars + 1, 1))
    float_output = np.zeros((num_xbars + 1, voltages.shape[0], conductances.shape[1]))
    
    voltages = voltages - total_min_v
    for i in range(num_xbars): 
        assert (np.min(voltages) >= 0), "The input voltage should be non-negative"
        # Compute the interval width for the current crossbar
        input_interval_widths[i, :] = (np.max(voltages) - np.min(voltages)) / num_steps if i == 0 else input_interval_widths[i - 1, :] / num_steps
        
        (qtz_input[i, :, :], 
         qtz_input_index[i, :, :], 
         a[i, :], b[i, :]) = Quantize_input(voltages, v_range, input_interval_widths[i, :], num_steps)
        
        # Compute the float output for the current crossbar
        float_output[i, :, :] = np.matmul(qtz_input_index[i, :, :] * input_interval_widths[i, :], conductances)
        
        # Compute the output vector and quantize it to the range [0, 255]
        out = np.matmul(qtz_input[i, :, :], qtz_conductances)
        min_out[i, :], max_out[i, :] = np.min(out), np.max(out)
        
        codes_out = np.arange(0, 256, dtype=int)
        qtz_output[i, :, :] = codes_out[
            np.clip(
                np.floor((out - min_out[i, :]) / (max_out[i, :] - min_out[i, :]) * 255).astype(int), 
                0, 255)]
        (qtz_output_deduct[i, :, :], 
         rescale_factor[i, :], rescale_bias[i, :]) = Deduct_VM(qtz_output[i, :, :], 
                                                               a[i, :], b[i, :], c, d, 
                                                               max_out[i, :], min_out[i, :], 
                                                               qtz_input_index[i, :, :] * input_interval_widths[i, :], 
                                                               conductances)
        
        # Update the voltages for the next crossbar
        voltages = voltages - input_interval_widths[i, :] * qtz_input_index[i, :, :]
    
    # Compute the VMM for the total_min_v uniform vector and put it in the last crossbar
    i = num_xbars
    total_min_v_qtz_vec = np.ones((1, voltages.shape[1])) * v_range[0]
    total_min_v_vec = np.ones((1, voltages.shape[1])) * total_min_v
    out = np.matmul(total_min_v_qtz_vec, qtz_conductances)
    float_output[i, :, :] = np.matmul(total_min_v_vec, conductances)
    min_out[i, :], max_out[i, :] = np.min(out), np.max(out)
    
    codes_out = np.arange(0, 256, dtype=int)
    qtz_output[i, :, :] = codes_out[
        np.clip(
            np.floor((out - min_out[i, :]) / (max_out[i, :] - min_out[i, :]) * 255).astype(int), 
            0, 255)]
    a[i, :], b[i, :] = 1, v_range[0] - total_min_v
    (qtz_output_deduct[i, :, :],
     rescale_factor[i, :], rescale_bias[i, :]) = Deduct_VM(qtz_output[i, :, :], 
                                                           a[i, :], b[i, :], c, d, 
                                                           max_out[i, :], min_out[i, :], 
                                                           total_min_v_qtz_vec, 
                                                           conductances)
    
    return qtz_input, qtz_input_index, input_interval_widths, \
           qtz_conductances, total_min_v, a, c, rescale_factor, rescale_bias, \
           qtz_output, qtz_output_deduct, float_output

'''
This function is to reconstruct the input voltage vectors and the output results from the quantized input and output
recover_input = total_min_v + sum(qtz_input_index[i, :, :] * input_interval_widths[i, :])
recover_output = sum(qtz_output[i, :, :] * input_interval_widths[i, :]) + total_min_v_output
    total_min_v_output is the last array in the qtz_output array
'''
def Reconstruct_output(qtz_input, qtz_input_index, input_interval_widths, 
                       total_min_v, qtz_output, a, c, rescale_factor, rescale_bias):
    '''
    Parameters:
        qtz_input: quantized input voltage vectors (num_xbars x N x l, [v_range[0], v_range[1]]), where N is the number of vectors and l is the length of each vector
        qtz_input_index: the indices of the quantized input voltage vectors (Nxl)
        input_interval_widths: the width of each interval in the quantization (num_xbars x 1)
        qtz_output: quantized output vector (Nxm), where N is the number of vectors and m is the output vector length`
    Returns:
        recover_input: the reconstructed input voltage vectors (Nxl)
        recover_output: the reconstructed output vector (Nxm)
    '''
    num_xbars = qtz_input.shape[0]
    recover_input = np.sum(qtz_input_index * input_interval_widths.reshape(-1, 1, 1), axis=0) + total_min_v
    
    recover_output = np.zeros(qtz_output.shape[1:])
    for i in range(num_xbars):
        recover_output += (qtz_output[i, :, :] * rescale_factor[i, :] + rescale_bias[i, :]) / (a[i] * c)
    # recover_output += (qtz_output[-1, :, :] * rescale_factor[-1, :] + rescale_bias[-1, :]) / c
    
    return recover_input, recover_output


def Compare_qtz_with_flout_out(qtz_output, float_output, c):
    '''
    Parameters:
        qtz_output: quantized output vector (Nxm), where N is the number of vectors and m is the output vector length
        float_output: the float output vector (Nxm)
        v_range: voltage quantization range (2x1), where the first element is the minimum and the second element is the maximum
    Returns:
        qtz_output: quantized output vector (Nxm), where N is the number of vectors and m is the output vector length
    '''
    num_xbars = qtz_output.shape[0]
    qtz_outs = np.zeros((num_xbars, qtz_output.shape[1], qtz_output.shape[2]))
    for i in range(num_xbars):
        current_qtz_output = qtz_output[i, :, :] / c
        current_float_output = float_output[i, :, :]
        # Compute the qtz output of the current float output
        min_out, max_out = np.min(current_float_output), np.max(current_float_output)
        codes_out = np.arange(0, 256, dtype=int)
        qtz_out = codes_out[
            np.clip(
                np.floor((current_float_output - min_out) / (max_out - min_out) * 255).astype(int), 
                0, 255)]
        # Compare the "current_qtz_output" with "qtz_out"
        # assert np.allclose(current_qtz_output, qtz_out), "The quantized output is not correct"
        qtz_outs[i, :, :] = qtz_out
        # Plot the current_float_output and current_qtz_output and qtz_out in 3 subplots
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax1.plot(np.arange(current_float_output.shape[1]), current_float_output[0], 'r', label='Float Output')
        plt.legend()
        plt.grid(True)
        
        ax2 = fig.add_subplot(312)
        ax2.plot(np.arange(current_qtz_output.shape[1]), current_qtz_output[0], 'b', label='Quantized Output')
        plt.legend()
        plt.grid(True)
        
        ax3 = fig.add_subplot(313)
        ax3.plot(np.arange(qtz_out.shape[1]), qtz_out[0], 'g', label='Quantized Output from float')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/Quantized_output_comparison.png')
        
    return qtz_outs


if __name__ == '__main__':
    # Example usage
    # Input, random generate voltages in a given range
    voltages = np.random.uniform(-1, 1, (2, 32))
    conductances = np.random.uniform(-1, 1, (32, 5))     # np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
    total_float_output = np.matmul(voltages, conductances)
    
    v_range = np.array([50, 240])
    g_range = np.array([50, 240])
    num_xbars = 2
    num_steps = 16
    
    # VMM with multiple crossbars
    (qtz_input, qtz_input_index, input_interval_widths, \
     qtz_conductances, total_min_v, a, c, rescale_factor, rescale_bias,\
     qtz_output, qtz_output_deduct, float_outputs) = VMM_with_multi_XB(voltages, conductances, v_range, g_range, num_xbars, num_steps)
    
    true_qtz_out = Compare_qtz_with_flout_out(qtz_output_deduct, float_outputs, c)
    
    recover_input, recover_output = Reconstruct_output(qtz_input, qtz_input_index, input_interval_widths, 
                                                       total_min_v, qtz_output_deduct, a, c, rescale_factor, rescale_bias)
    
    # Plot float VMM output and reconstructed output in different subplots
    row_to_plot = 0
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(np.arange(recover_output.shape[1]), recover_output[row_to_plot], 'r', label='Reconstructed Output')
    plt.legend()
    plt.grid(True)
    
    ax2 = fig.add_subplot(212)
    ax2.plot(np.arange(total_float_output.shape[1]), total_float_output[row_to_plot], 'b', label='Float Output')
    #save the figure
    plt.legend()
    plt.grid(True)
    plt.savefig('results/VMM_with_multi_XB.png')
    
    # Plot the float input and reconstructed input in a single plot
    fig = plt.figure()
    plt.plot(np.arange(recover_input.shape[1]), recover_input[row_to_plot], 'r', label='Reconstructed Input')
    plt.plot(np.arange(voltages.shape[1]), voltages[row_to_plot], 'b', label='Float Input')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/Input_reconstruction.png')
    print("Done")
    
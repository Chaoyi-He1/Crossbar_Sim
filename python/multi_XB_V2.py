import numpy as np
import sys
from visualize import plot_array


def residue_coefficients(residue_mtx, g_range):
    '''
    The residue_mtx is the residue between the target conductance matrix and the sum of the conductance matrices of the previous levels
        residue_mtx = target_conductance_matrix - sum(residue_coefficients[i] * real_conductance_matrices[i]) for i in previous levels
    The residue_coefficients will be calculated based on the residue_mtx and g_range, 
    make the residue_mtx to be within the range of g_range after multiplied by the residue_coefficient:
        
        1. extract the sign of the residue_mtx
        2. if the residue_mtx is less than 1000 times of g_range[0], the residue_coefficient is 1
        3. if the residue_mtx is out of the range of g_range, calculate the residue_coefficient based on the g_range and residue_mtx
            3.1 if the abs value of residue_mtx[i, j] > g_range[1], the residue_coefficient[i, j] = 1/(ceil(abs(residue_mtx[i, j])/g_range[1])) * sign(residue_mtx[i, j])
            3.2 if the abs value of residue_mtx[i, j] < g_range[0], the residue_coefficient[i, j] = floor(abs(residue_mtx[i, j])/g_range[0]) * sign(residue_mtx[i, j])
        4. if the residue_mtx is within the range of g_range, the residue_coefficient is +-1, based on the sign of the residue_mtx
    '''
    residue_coefficient = np.zeros(residue_mtx.shape)
    residue_sign = np.sign(residue_mtx)
    residue_abs = np.abs(residue_mtx)
    residue_coefficient[residue_abs < g_range[0]] = residue_sign[residue_abs < g_range[0]] * np.floor(residue_abs[residue_abs < g_range[0]] / g_range[0])
    residue_coefficient[residue_abs > g_range[1]] = residue_sign[residue_abs > g_range[1]] / np.ceil(residue_abs[residue_abs > g_range[1]] / g_range[1])
    residue_coefficient[(residue_abs >= g_range[0]) & (residue_abs <= g_range[1])] = residue_sign[(residue_abs >= g_range[0]) & (residue_abs <= g_range[1])]
    residue_coefficient[residue_abs < g_range[0] / 1000] = 1
    return residue_coefficient


def residue_conductance_mtx(residue_mtx, residue_coefficients, g_range):
    '''
    Calculate the residue conductance matrix for this level's crossbar programming based on the residue_mtx and residue_coefficients
    The residue conductance matrix needs to be within the range of g_range after calculation:
        residue_conductance_matrix = residue_mtx / residue_coefficients
    if the residue_conductance_matrix is out of the range of g_range, raise an error
    '''
    residue_conductance_matrix = residue_mtx / residue_coefficients
    if np.any(residue_conductance_matrix < g_range[0]) or np.any(residue_conductance_matrix > g_range[1]):
        raise ValueError('The residue conductance matrix is out of the range of g_range')
    return residue_conductance_matrix


def Quantize_mtx(input_mtx, g_range, num_levels=16):
    '''
    Parameters:
        input_mtx: input conductance matrix (m x n)
        g_range: conductance quantization range (2x1), where the first element is the minimum and the second element is the maximum
                  The conductance will be quantized to integers in the range [g_range[0], g_range[1]]
    Returns:
        qtz_mtx: quantized input conductance matrix (m x n)
        qtz_indices: the indices of the quantized input conductance matrix (m x n)
    '''
    # Determine codes and prepare for broadcasting
    codes = np.linspace(g_range[0], g_range[1], num_levels, endpoint=False, dtype=int)
    
    # Compute min and max for each vector
    min_vals = np.min(input_mtx)
    max_vals = np.max(input_mtx)
    
    # Calculate interval widths for each vector
    interval_widths = (max_vals - min_vals) / num_levels
    
    # Normalize vectors to range [0, 1] within their respective min-max range
    normalized_mtx = (input_mtx - min_vals) / (max_vals - min_vals)
    
    # Scale normalized vectors to code indices and clip to valid range
    scaled_indices = np.floor(normalized_mtx * num_levels).astype(int)
    scaled_indices = np.clip(scaled_indices, 0, num_levels - 1)
    
    # Map indices to quantized voltage values
    quantized_mtx = codes[scaled_indices]
    
    return quantized_mtx, scaled_indices, interval_widths



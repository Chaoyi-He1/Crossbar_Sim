import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt


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
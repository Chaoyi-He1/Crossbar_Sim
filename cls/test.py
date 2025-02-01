import time
import os
import datetime
import random
import math
import torch
import copy
from pathlib import Path

import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    gt_label_path = "/data/chaoyi_he/Crossbar_Sim/cls/nn_labels.npy"
    pred_label_path = "/data/chaoyi_he/Crossbar_Sim/cls/fir_mlp_out.npy"
    
    gt_label = np.load(gt_label_path)
    pred_label = np.load(pred_label_path)
    pred_label = np.argmax(pred_label, axis=1)
    
    # create confusion matrix
    num_classes = max(gt_label) + 1
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(gt_label)):
        confusion_matrix[gt_label[i], pred_label[i]] += 1

    # save confusion matrix
    np.save("/data/chaoyi_he/Crossbar_Sim/cls/confusion_matrix.npy", confusion_matrix)
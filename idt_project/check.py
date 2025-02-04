'''
check if every data in "/data/chaoyi_he/Crossbar_Sim/idt_project/data/Test/idt_test_data_wifi.npy" and its label is in
/data/chaoyi_he/Crossbar_Sim/idt_project/data/Train/idt_train_data_wifi_whole.npy
'''
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    test_data = np.load("/data/chaoyi_he/Crossbar_Sim/idt_project/data/Test/idt_test_data_wifi.npy")
    train_data = np.load("/data/chaoyi_he/Crossbar_Sim/idt_project/data/Train/idt_train_data_wifi_whole.npy")
    test_label = np.load("/data/chaoyi_he/Crossbar_Sim/idt_project/data/Test/idt_test_label_wifi.npy")
    train_label = np.load("/data/chaoyi_he/Crossbar_Sim/idt_project/data/Train/idt_train_label_wifi_whole.npy")
    
    '''
    data in test_data is shape (N, 16, 16, 2)
    train_data is shape (M, 16, 16, 2)
    check if every (16, 16, 2) in test_data is in train_data
    and check if the label is the same
    '''
    for i in tqdm(range(test_data.shape[0])):
        if not np.any(np.all(test_data[i] == train_data, axis=(1, 2, 3))):
            print("test data not in train data")
            break
        else:
            if train_label[np.argmax(np.all(test_data[i] == train_data, axis=(1, 2, 3)))] != test_label[i]:
                print("test label not in train label")
                break
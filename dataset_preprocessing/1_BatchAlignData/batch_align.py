
import os
import pandas as pd
import numpy as np
import cupy as cp
from scipy.signal import correlate


def read_data(lsl_file, vicon_file, vicon_extended):
    lsl_data = pd.read_xdf(lsl_file)
    vicon_data = pd.read_csv(vicon_file)
    vicon_extended_data = pd.read_csv(vicon_extended)

    return lsl_data, vicon_data, vicon_extended_data

def align_data(lsl_data, vicon_data):
    cross_corr = correlate(lsl_data, vicon_data, mode='full')
    lag = np.argmax(cross_corr) - len(lsl_data) + 1
    aligned_vicon_data = np.roll(vicon_data, -lag)

    return aligned_vicon_data, lag

def align_all_data(xdf_folder, csv_folder, reference_event):
    aligned_data_list = []

    for xdf_subfolder, csv_subfolder in zip(os.listdir(xdf_folder), os.listdir(csv_folder)):
        xdf_file_path = os.path.join(xdf_folder, xdf_subfolder, 'your_file.xdf')  # Replace 'your_file.xdf' with actual file name
        csv_file_path = os.path.join(csv_folder, csv_subfolder, 'your_file.csv')  # Replace 'your_file.csv' with actual file name

        aligned_data = align_data(xdf_file_path, csv_file_path, reference_event)
        aligned_data_list.append(aligned_data)

    return aligned_data_list

# Example usage
lsl_folder_path = 'path/to/lsl'
vicon_folder_path = 'path/to/vicon'
reference_event_name = 'your_reference_event'

aligned_data_list = align_all_data(xdf_folder_path, csv_folder_path, reference_event_name)

# Further processing with the aligned_data_list

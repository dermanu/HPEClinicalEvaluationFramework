import os
import numpy as np
import torch
import skeletonMorphing.readDatasetMorph as rdm

def all_participants(data_folder):
    """
    Function to return all participant folders in the dataset folder.
    """
    par = []
    par_num = []
    for root, dirs, files in os.walk(data_folder):
        for dir in dirs:
            if dir.startswith('par'):
                par.append(dir)
                par_num.append(int(dir[3:]))

    par = sorted(par)
    return par, par_num

def list_to_file_name(lst):
    """
    Function to convert a list to a string that can be used as a file name.
    """
    return '_'.join([str(i) for i in lst])


"""
Needed due to how multiprocessing works in windows (Ok without in Linux)
"""

def run_load(datapath: str):
    # Define the path to the folder containing the segmented data
    #data_folder = '/home/emanu/Desktop/SegmentedData'
    data_folder = datapath + '/segmented'

    # Assuming you have a model type
    model_type = 'mediapipe'

    cam = [0, 1, 2, 3, 4, 5]
    #cam = [0]

    # Assuming you have a list of movements
    mov = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    #mov = [1]

    # Assuming you have a list of participant folders
    #par = [14]
    par_full, par = all_participants(data_folder)
    par = [10, 26]

    # Read the dataset for the current participant
    for p in par:
        p = [p]
        my_dataset = rdm.ReadDatasetFiles(data_folder, p, mov, cam, model_type)

        # Save the dataset to a .pth file named after the participant

        print("Saving Dataset")
        file_name = f"{datapath}/morph_dataset/par_{list_to_file_name(p)}_{model_type}_dataset.pth"
        torch.save(my_dataset, file_name)


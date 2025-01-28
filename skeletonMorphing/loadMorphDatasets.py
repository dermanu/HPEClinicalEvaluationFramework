import os
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
    par_num = sorted(par_num)
    return par, par_num

def list_to_file_name(lst):
    """
    Function to convert a list to a string that can be used as a file name.
    """
    return '_'.join([str(i) for i in lst])

def run_load(datapath: str, model_type: str):
    """
    Method to load all data and store as pytorch datasets to file. Datapath should contain path to folder
    where /segmented folder is stored. Data will be stored in datapath under /morphed

    Parameters:
    - datapath: Datapath where data is segmented ground truth and HPE prediction/dataset is stored at
    - model_type: Defining model type so that the appropriate keypoint order can be used (move outside)
    """

    # Define the path to the folder containing the segmented data
    data_folder = datapath + '\segmented'

    # List of cameras angles and motions to be used (all)
    cam = [0, 1, 2, 3, 4, 5]
    mov = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    # Assuming you have a list of participant folders
    par_full, par = all_participants(data_folder)
    print(par)

    # Read the dataset for the current participant
    for p in par:
        if p in {12, 14, 15, 4, 19, 11, 23}:
            print("Skipping", p)
            continue
        p = [p]
        my_dataset = rdm.ReadDatasetFiles(data_folder, p, mov, cam, model_type)

        # Save the dataset to a .pth file named after the participant
        print("Saving Dataset")
        file_name = f"{datapath}\morph_dataset\par_{list_to_file_name(p)}_{model_type}_dataset.pth"
        torch.save(my_dataset, file_name)

if __name__ == '__main__':
    run_load('E:\MoCap', model_type = 'mediapipe')

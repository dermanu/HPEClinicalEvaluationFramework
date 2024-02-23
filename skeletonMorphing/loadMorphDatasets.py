import os
import torch
import readDatasetMorph as rdm

# Define the path to the folder containing the segmented data
data_folder = '/home/emanu/Desktop/SegmentedData'

# Assuming you have a list of participant folders
# par = [12, 14, 15]
par = [14]
# Assuming you have a list of cameras
cam = [0, 1, 2, 3, 4, 5]

# Assuming you have a list of movements
mov = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Assuming you have a model type
model_type = 'mediapipe'

# Read the dataset for the current participant
my_dataset = rdm.ReadDatasetFiles(data_folder, par, mov, cam, model_type)

# Save the dataset to a .pth file named after the participant
file_name = f"par_{par}_{model_type}_dataset.pth"
torch.save(my_dataset, file_name)
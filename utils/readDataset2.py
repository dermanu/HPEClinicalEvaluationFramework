import os
import pandas as pd
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import models.mediapipeMono as MediaPipe
import multiprocessing


class ReadDatasetFiles(Dataset):
    def __init__(self, data_dir, participant_list, movement_list, camera_list, model_type):
        self.data_dir = data_dir
        self.participant_list = participant_list
        self.movement_list = movement_list
        self.camera_list = camera_list
        self.model_type = model_type

        # Initialize these lists in the constructor
        self.csv_file_paths = self.read_dataset_path()
        self.datasets = self.create_datasets()

    def read_dataset_path(self):
        """
        Reads and processes CSV and AVI files in a dataset.

        Parameters:
        - dataset_folder (str): The root folder of the dataset.
        - participants (list): List of selected participants.
        - cameras (list): List of selected cameras.
        - movements (list): List of selected movements.
        """

        # Convert to strings and add prefixes
        participants = ['par' + str(participant) for participant in self.participant_list]
        cameras = ['_Cam' + str(camera) + "." for camera in self.camera_list]
        movements = ['_Mov' + str(movement) + '_' for movement in self.movement_list]

        # Lists to store file paths
        csv_file_paths = []
        avi_file_paths = []

        # Iterate through selected participants
        for participant in participants:
            participant_folder = os.path.join(self.data_dir, participant)

            # Iterate through files in participant's folder
            for file_name in os.listdir(participant_folder):
                file_path = os.path.join(participant_folder, file_name)

                # Check if the file is a CSV file
                if file_name.endswith('.csv') and any(camera in file_name for camera in cameras) and any(
                        movement in file_name for movement in movements):
                    csv_file_paths.append(file_path)

        print('All csv files paths are read')

        return csv_file_paths

    def create_datasets(self):
        datasets = []
        #i = 0

        # Use multiprocessing to parallelize dataset creation
        with multiprocessing.Pool() as pool:
            datasets = pool.map(self.create_single_dataset, self.csv_file_paths)


        #for csv_file_path in self.csv_file_paths:
        #    i += 1
        #    print('Getting dataset nr ' + str(i) + ' of ' + str(len(self.csv_file_paths)) + '...')
        #    dataset = SingleCSVFileDataset(csv_file_path, self.model_type)
        #    datasets.append(dataset)

        return datasets

    def create_single_dataset(self, csv_file_path):
        i = self.csv_file_paths.index(csv_file_path) + 1
        print('Getting dataset nr ' + str(i) + ' of ' + str(len(self.csv_file_paths)) + '...')
        dataset = SingleCSVFileDataset(csv_file_path, self.model_type)
        return dataset

    def __len__(self):
        total_length = sum(len(dataset) for dataset in self.datasets)
        return total_length

    def __getitem__(self, idx):
        # Determine which dataset the index corresponds to
        dataset_idx = 0
        while idx >= len(self.datasets[dataset_idx]):
            idx -= len(self.datasets[dataset_idx])
            dataset_idx += 1

        # Retrieve the item from the corresponding dataset
        return self.datasets[dataset_idx][idx]


class SingleCSVFileDataset(Dataset):
    def __init__(self, csv_file_path, model_type):
        self.csv_file_path = csv_file_path
        self.model_type = model_type

        # Add any additional initialization for your SingleCSVFileDataset
        self.csv_data = self.load_csv_data()
        self.pose_inf, self.confidences_inf = self.load_video_data()

    def load_csv_data(self):
        # Load CSV data
        csv_data = pd.read_csv(self.csv_file_path)

        # Drop irrelevant columns
        csv_data.drop(columns=['Time', 'CameraFrame', 'Iteration'], inplace=True)

        # Align the columns (you may need to modify this part based on your needs)
        csv_data = self.align_keypoints(csv_data)
        csv_data = np.array(list(csv_data.values()))
        csv_data = csv_data.transpose((1, 0, 2))

        return csv_data

    def align_keypoints(self, keypoints_org):
        """
        Aligns keypoints in a CSV file based on different alignment identifiers.

        Parameters:
        - keypoints_org (pd.DataFrame or None): Original DataFrame containing the keypoints.
        - model_name (str): String identifier for selecting the alignment array based on the respective model.

        Returns:
        - aligned_df (pd.DataFrame or None): Aligned DataFrame if output_path is None, else None.
        """

        # Define different order arrays based on alignment_identifier
        keypoints_org_names = ['RBAK', 'LTOE', 'LASI', 'CLAV', 'T10', 'LPSI', 'RWJC', 'RASI', 'RTIB', 'LANK', 'LAJC',
                               'LKJC', 'RKJC', 'RFRM', 'C7', 'RWRA', 'LEJC', 'LWJC', 'LFRM', 'LWRB', 'LTHI', 'RTHI',
                               'RSJC', 'RAJC', 'LWRA', 'LSHO', 'RHEE', 'STRN', 'RPSI', 'LELB', 'LUPA', 'RWRB', 'RTOE',
                               'LKNE', 'RSHO', 'RHJC', 'RANK', 'RKNE', 'LSJC', 'LHEE', 'RELB', 'RUPA', 'REJC', 'LTIB',
                               'LHJC']


        if self.model_type == 'mediapipe':
            column_mapping = {
                'RShoulder': 'RSJC', # 12
                'LShoulder': 'LSJC', # 11
                'RElbow': 'REJC', # 14
                'LElbow': 'LEJC', # 13
                'RWrist': 'RWJC', # 16
                'LWrist': 'LWJC', # 15
                'RHip': 'RHJC', # 24
                'LHip': 'LHJC', # 23
                'RKnee': 'RKJC', # 26
                'LKnee': 'LKJC', # 25
                'RAnkle': 'RAJC', # 28
                'LAnkle': 'LAJC', # 27
                'RHeel': 'RHEE', # 30
                'LHeel': 'LHEE', # 29
                'RFootIndex': 'RTOE', # 32
                'LFootIndex': 'LTOE', # 31
            }
            self.selected_columns = [12, 11, 14, 13, 16, 15, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31]

        elif self.model_type == 'openpose':
            self.column_mapping = [1, 3, 2]
            self.selected_columns = [12, 11, 14, 13, 16, 15, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31]
        else:
            raise ValueError(f"Invalid model_name: {self.model_type }")

        # Extract original keypoint names (assuming they follow a pattern)
        keypoints_org_names = set(col.rsplit('_', 1)[0] for col in keypoints_org.columns)

        # Create a dictionary with sublists (X, Y, Z) of the original keypoints
        keypoints_org_subarrays = {}

        # Group X, Y, Z coordinates for each joint
        for keypoint in keypoints_org_names:
            keypoint_cols = [f"{keypoint}_{coord}" for coord in ['X', 'Y', 'Z']]
            keypoints_org_subarrays[keypoint] = keypoints_org[keypoint_cols].to_numpy()

        # Filter columns based on column_mapping and reorder the DataFrame
        ordered_keypoints_org_subarrays = dict(zip(column_mapping.keys(), [keypoints_org_subarrays[key] for key in column_mapping.values()]))

        return ordered_keypoints_org_subarrays

    def load_video_data(self):
        # Load video data
        print('Start loading video data' + self.csv_file_path.replace('.csv', '.avi') + '...')
        avi_file_path = self.csv_file_path.replace('.csv', '.avi')
        cap = cv2.VideoCapture(avi_file_path)

        if self.model_type == 'mediapipe':
            pose_keypoints = MediaPipe.inference_video(cap)
            confidences = pose_keypoints[0][:, self.selected_columns, 0]
            pose_keypoints = pose_keypoints[0][:, self.selected_columns, 1:]
        else:
            raise ValueError(f"Invalid model_name: {self.model_type}")
        print('Finished loading video data' + self.csv_file_path.replace('.csv', '.avi') + '...')

        return pose_keypoints, confidences

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        # Combine CSV and video data into a single dictionary
        combined_data = {'pose_gt': self.csv_data[idx], 'pose_inf': self.pose_inf[idx],
                         'confidences_inf': self.confidences_inf[idx]}

        return combined_data
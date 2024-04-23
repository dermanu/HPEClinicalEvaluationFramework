import os
import sys
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
import models.mediapipeMono as MediaPipe
import multiprocessing


class ReadDatasetFiles(Dataset):
    def __init__(self, data_dir, participant_list, movement_list, camera_list, model_type, init=True):
        self.data_dir = data_dir
        self.participant_list = participant_list
        self.movement_list = movement_list
        self.camera_list = camera_list
        self.model_type = model_type

        # Initialize these lists in the constructor
        self.csv_file_paths = []
        self.datasets = []
        if init:
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
        #cameras = ['_Cam' + str(camera) + "." for camera in self.camera_list]
        cameras = ['_Cam' + str(camera) + "." for camera in [0, 0, 0, 0, 0, 0]]
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


                    if file_path in csv_file_paths:
                        print(file_path)
                        continue

                    csv_file_paths.append(file_path)

        print('All csv files paths are read')

        if len(csv_file_paths) == 0:
            raise ValueError('No CSV files found in the specified folder')

        return csv_file_paths

    def create_datasets(self):
        #datasets = []
        #i = 0

        # Use multiprocessing to parallelize dataset creation
        with multiprocessing.Pool(processes=10) as pool:
            datasets = pool.map(self.create_single_dataset, self.csv_file_paths)

        
        print('All datasets are created')
        return datasets

    def create_single_dataset(self, csv_file_path):
        i = self.csv_file_paths.index(csv_file_path) + 1
        print('Getting dataset nr ' + str(i) + ' of ' + str(len(self.csv_file_paths)) + '...', flush=True)
        dataset = SingleCSVFileDataset(csv_file_path, self.model_type)
        return dataset

    def _create_self_copy_new_dataset(self, datasets):
        read_dataset_files = ReadDatasetFiles(self.data_dir, self.participant_list, self.movement_list, self.camera_list, self.model_type, init=False)
        read_dataset_files.datasets = datasets
        read_dataset_files.csv_file_paths = self.csv_file_paths
        return read_dataset_files
    
    def get_single_train(self, dataset):
        return dataset.get_dataset(train=True)

    def get_single_test(self, dataset):
        return dataset.get_dataset(train=False)


    def get_train_test(self):
        """
        Using threads, get train and test data stored in all SingleCSVFileDataset in this class
        :return:
        """

        with multiprocessing.Pool(processes=10) as pool:
            train = pool.map(self.get_single_train, self.datasets)
            test = pool.map(self.get_single_test, self.datasets)

        train_dataset = self._create_self_copy_new_dataset(train)
        test_dataset = self._create_self_copy_new_dataset(test)

        return train_dataset, test_dataset

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

    def __str__(self):
        return f"ReadDatasetFiles({self.data_dir}), len={len(self)}"


class SingleCSVFileDataset(Dataset):
    def __init__(self, csv_file_path, model_type, init=True):
        self.csv_file_path = csv_file_path
        self.model_type = model_type
        self.csv_data = None
        self.pose_inf = None
        self.confidences_inf = None
        self.selected_columns = None

        # Add any additional initialization for your SingleCSVFileDataset
        if init:
            self.csv_data, self.train_frames, self.test_frames = self.load_csv_data()
            self.pose_inf, self.confidences_inf = self.load_video_data()

    def get_camera(self):
        return int(self.csv_file_path[-5])

    def _get_csv_data(self, csv_data : pd.DataFrame):
        """
        Aligns the columns of the CSV data based on the model type.
        """

        csv_data = self.align_keypoints(csv_data)
        csv_data = np.array(list(csv_data.values()))
        csv_data = csv_data.transpose((1, 0, 2))

        return csv_data

    def get_dataset(self, train=True):
        """
        Get dataset from csv path
        Creates a new SingleCSVFileDataset (init false to not load data from file again)
        Sets values of the new dataset if it should use train or test data from the real dataset
        :param train:
        :return:
        """
        dataset = SingleCSVFileDataset(self.csv_file_path, self.model_type, init=False)
        csv_data, pose_inf, confidences_inf = self.get_training_data() if train else self.get_test_data()
        dataset.csv_data = csv_data
        dataset.pose_inf = pose_inf
        dataset.confidences_inf = confidences_inf
        return dataset

    def get_training_data(self):
        return self.csv_data[self.train_frames], self.pose_inf[self.train_frames], self.confidences_inf[self.train_frames]

    def get_test_data(self):
        return self.csv_data[self.test_frames], self.pose_inf[self.test_frames], self.confidences_inf[self.test_frames]

    def get_train_test_datasets(self):
        training = self.get_dataset(train=True)
        test = self.get_dataset(train=False)
        return training, test

    def get_split_indexes(self, csv_data, split=[0.8,0.2]):
        """
        Splits data into train/test based on 80/20 split and keeps track by indexes stored in class
        Splits repetitions of movements into 80/20
        :param csv_data:
        :param split:
        :return:
        """
        # Drop NaN values in Iteration column
        iteration_values = csv_data['Iteration'].dropna().unique()

        # Perform 80/20 split
        split_index = int(len(iteration_values) * split[0])

        # If Train take 80% of the data, else take 20% (Must find better split)
        values = iteration_values[:split_index]
        

        # Fill NaN values in Iteration column
        csv_data['Iteration'] = csv_data['Iteration'].ffill()

        # Create a new column to filter based on train
        csv_data['Training'] = csv_data['Iteration'].isin(values)

        # Filter based on train to return test dataset if not train
        train_index = csv_data[csv_data['Training'] == 1].index
        test_index = csv_data[csv_data['Training'] == 0].index
        csv_data.drop(columns=['Training'], inplace=True)

        assert len(train_index) + len(test_index) == len(csv_data), "Train and test indexes do not match the length of the dataset"

        return train_index, test_index

    def load_csv_data(self):
        # Load CSV data
        data = []
        train_index = None
        #for i in range(6):
        path = self.csv_file_path
        #path.replace(f"Cam{(i-1) if i > 0 else 0}", f"Cam{i}")
        #path[-5] = str(i)
        csv_data = pd.read_csv(path)
        print(path)

        if train_index is None:
            train_index, test_index = self.get_split_indexes(csv_data)

        # Drop irrelevant columns
        csv_data.drop(columns=['Time', 'CameraFrame', 'Iteration'], inplace=True)

        # Align the columns (you may need to modify this part based on your needs)
        csv_data = self._get_csv_data(csv_data)
        #data.append(csv_data)

        #csv_data = np.array(data)
        #csv_data = np.swapaxes(csv_data, 0, 1)

        print(csv_data.shape)

        return csv_data, train_index, test_index

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
                'RShoulder': 'RSJC', # 12 - 0
                'LShoulder': 'LSJC', # 11 - 1
                'RElbow': 'REJC', # 14 - 2
                'LElbow': 'LEJC', # 13 - 3
                'RWrist': 'RWJC', # 16 - 4
                'LWrist': 'LWJC', # 15 - 5
                'RHip': 'RHJC', # 24 - 6
                'LHip': 'LHJC', # 23 - 7
                'RKnee': 'RKJC', # 26 - 8
                'LKnee': 'LKJC', # 25  - 9
                'RAnkle': 'RAJC', # 28 - 10
                'LAnkle': 'LAJC', # 27 - 11
                'RHeel': 'RHEE', # 30 - 12
                'LHeel': 'LHEE', # 29 - 13
                'RFootIndex': 'RTOE', # 32 - 14
                'LFootIndex': 'LTOE', # 31 - 15
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
        data_key = []
        data_conf = []
        avi_file_path = self.csv_file_path.replace('.csv', '.avi')
        for i in range(6):


            avi_file_path = avi_file_path.replace(f"Cam{(i-1) if i > 0 else 0}", f"Cam{i}")
            print('Start loading video data' + avi_file_path + '...')
            cap = cv2.VideoCapture(avi_file_path)

            if self.model_type == 'mediapipe':

                pose_keypoints = MediaPipe.inference_video(cap)
                confidences = pose_keypoints[0][:, self.selected_columns, 0]
                pose_keypoints = pose_keypoints[0][:, self.selected_columns, 1:]
                #category = np.full((pose_keypoints.shape[0], 1), self.get_camera())
                #one_hot = np.eye(6)[category.squeeze()]
                #one_hot = np.expand_dims(one_hot, axis = 1)
                #one_hot = np.repeat(one_hot, 16, axis=1)
                #pose_keypoints = np.concatenate([pose_keypoints, one_hot], axis = -1)
                #print(one_hot)
            else:
                raise ValueError(f"Invalid model_name: {self.model_type}")
            print('Finished loading video data' + avi_file_path + '...')

            data_key.append(pose_keypoints)
            data_conf.append(confidences)

        pose_keypoints = np.array(data_key)
        confidences = np.array(data_conf)
        pose_keypoints = np.swapaxes(pose_keypoints, 0, 1)
        confidences = np.swapaxes(confidences, 0, 1)
        print(pose_keypoints.shape, confidences.shape)
        return pose_keypoints, confidences

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        # Get the data for the specified index
        csv_data = self.csv_data[idx]
        pose_inf = self.pose_inf[idx]
        confidences_inf = self.confidences_inf[idx]

        # Check for NaN values in the CSV data
        if np.isnan(csv_data).any() or np.isnan(pose_inf).any() or np.isnan(confidences_inf).any():
            # If NaN values are present, return None to skip this sample
            return None

        # Combine CSV and video data into a single dictionary
        combined_data = {'pose_gt': self.csv_data[idx], 'pose_inf': self.pose_inf[idx],
                         'confidences_inf': self.confidences_inf[idx]}

        return combined_data

    def __str__(self):
        return f"SingleCSVFileDataset({self.csv_file_path}), len={len(self)}"


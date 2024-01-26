import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset


class ReadDatasetFiles(Dataset):
    def __init__(self, data_dir, participant_list, movement_list, camera_list, model_type):
        self.data_dir = data_dir
        self.participant_list = participant_list
        self.movement_list = movement_list
        self.camera_list = camera_list
        self.model_type = model_type

        # Initialize these lists in the constructor
        self.csv_file_paths, self.avi_file_paths = self.read_dataset_path()

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

                # Check if the file is an AVI file
                elif file_name.endswith('.avi') and any(camera in file_name for camera in cameras) and any(
                        movement in file_name for movement in movements):
                    avi_file_paths.append(file_path)

        return csv_file_paths, avi_file_paths

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
        keypoints_org_names = ['RBAK', 'LTOE', 'LASI', 'CLAV', 'T10', 'LPSI', 'RWJC', 'RASI', 'RTIB', 'LANK', 'LAJC', 'LKJC',
                         'RKJC', 'RFRM', 'C7', 'RWRA', 'LEJC', 'LWJC', 'LFRM', 'LWRB', 'LTHI', 'RTHI', 'RSJC', 'RAJC',
                         'LWRA', 'LSHO', 'RHEE', 'STRN', 'RPSI', 'LELB', 'LUPA', 'RWRB', 'RTOE', 'LKNE', 'RSHO', 'RHJC',
                         'RANK', 'RKNE', 'LSJC', 'LHEE', 'RELB', 'RUPA', 'REJC', 'LTIB', 'LHJC']

        keypoints_coco_names = ['Nose', 'REyeIn', 'REye', 'REyeOut', 'LEyeIn', 'LEye', 'LEyeOut', 'REar', 'LEar', 'RMouth',
                          'LMouth',
                          'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist',
                          'RPinky', 'LPinky', 'RIndex', 'LIndex', 'RThumb', 'LThumb', 'RHip', 'LHip',
                          'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RHeel', 'LHeel', 'RFootIndex', 'LFootIndex']

        if self.model_type == 'openpose':
            column_mapping = {
                'RShoulder': 'RSJC',
                'LShoulder': 'LSJC',
                'RElbow': 'REJC',
                'LElbow': 'LEJC',
                'RWrist': 'RWJC',
                'LWrist': 'LWJC',
                'RHip': 'RHJC',
                'LHip': 'LHJC',
                'RKnee': 'RKJC',
                'LKnee': 'LKJC',
                'RAnkle': 'RAJC',
                'LAnkle': 'LAJC',
                'RHeel': 'RHEE',
                'LHeel': 'LHEE',
                'RFootIndex': 'RTOE',
                'LFootIndex': 'LTOE',
            }
        elif self.model_type  == 'mediapipe':
            order_array = [1, 3, 2]
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
        df = keypoints_org_subarrays[column_mapping.keys()]
        df.columns = [column_mapping[col] for col in df.columns if col in column_mapping]

        # Create a DataFrame with the order_array
        order_df = pd.DataFrame({'Order': order_array})

        # Merge or sort keypoints based on the order_array
        merged_df = pd.merge(order_df, keypoints_org, left_on='Order', right_index=True)

        return merged_df

    def __len__(self):
        return len(self.csv_file_paths)

    def __getitem__(self, idx):
        # Load CSV data
        csv_data = pd.read_csv(self.csv_file_paths[idx])
        # Drop irrelevant columns
        csv_data.drop(columns=['Time', 'CameraFrame', 'Iteration'], inplace=True)
        # Align the columns
        csv_data = self.align_keypoints(csv_data)

        # Load video data
        video_path = self.avi_file_paths[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Your video frame preprocessing logic here
            # Convert the frame to tensor if needed
            frame = torch.from_numpy(frame)
            frames.append(frame)
        cap.release()
        video_data = torch.stack(frames)

        # Your additional preprocessing logic here

        return {'csv_data': csv_data, 'video_data': video_data}

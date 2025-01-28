import os
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
import models.mediapipeMono as MediaPipe
import multiprocessing
import traceback


class ReadDatasetFiles(Dataset):
    """
    A class for managing and loading datasets for training the morphing model.
    """

    def __init__(self, data_dir, participant_list, movement_list, camera_list, model_type, init=True):
        """
        Initializes the dataset class with the specified parameters.

        Parameters:
         - data_dir (str): Path to the root dataset directory.
         - participant_list (list): List of participants to include.
         - movement_list (list): List of movements to include.
         - camera_list (list): List of cameras to include.
         - model_type (str): Model type ('mediapipe', 'custom other HPE model') for keypoint alignment.
         - init (bool): Whether to immediately read dataset paths and create datasets.
        """

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
        Reads the file paths for the CSV files matching the participants, cameras, and movements.

        Returns:
        - csv_file_paths (list): List of valid CSV file paths.
        """
        # Convert to strings and add prefixes
        participants = ['par' + str(participant) for participant in self.participant_list]
        cameras = ['_Cam' + str(camera) + "." for camera in [0, 0, 0, 0, 0, 0]]
        movements = ['_Mov' + str(movement) + '_' for movement in self.movement_list]

        # Lists to store file paths
        csv_file_paths = []

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
        """
        Creates datasets for each CSV file path using multiprocessing.

        Returns:
        - datasets (list): List of datasets created.
        """
        with multiprocessing.Pool(processes=10) as pool:
            # Use a partial function to pass additional arguments to the worker function
            results = pool.map(self.create_single_dataset_safe, self.csv_file_paths)

        # Filter out any None results which indicate a failed dataset creation
        datasets = [result for result in results if result is not None]

        print('All datasets are created')
        return datasets


    def create_single_dataset_safe(self, csv_file_path):
        """
        Safely creates a dataset for a single CSV file, allowing for an warning if a csv file can not be read.

        Parameters:
        - csv_file_path (str): Path to the CSV file.

        Returns:
        - dataset (SingleCSVFileDataset or None): Dataset created, or None if an error occurred.
        """
        try:
            return self.create_single_dataset(csv_file_path)
        except Exception as e:
            i = self.csv_file_paths.index(csv_file_path) + 1
            print(f"Error creating dataset {i} of {len(self.csv_file_paths)}: {e}")
            traceback.print_exc()
            return None


    def create_single_dataset(self, csv_file_path):
        """
        Creates a dataset object for a single CSV file.

        Parameters:
        - csv_file_path (str): Path to the CSV file.

        Returns:
        - dataset (SingleCSVFileDataset): Dataset object.
        """
        i = self.csv_file_paths.index(csv_file_path) + 1
        print(f'Getting dataset nr {i} of {len(self.csv_file_paths)}...', flush=True)
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
        Splits the datasets into training and testing sets using multiprocessing.

        Returns:
        - train_dataset (ReadDatasetFiles): Dataset object containing the training data.
        - test_dataset (ReadDatasetFiles): Dataset object containing the testing data.
        """
        with multiprocessing.Pool(processes=10) as pool:
            train = pool.map(self.get_single_train, self.datasets)
            test = pool.map(self.get_single_test, self.datasets)

        train_dataset = self._create_self_copy_new_dataset(train)
        test_dataset = self._create_self_copy_new_dataset(test)

        return train_dataset, test_dataset


    def __len__(self):
        """
        Returns the total number of samples across all datasets.

        Returns:
        - total_length (int): Total number of samples.
        """
        total_length = sum(len(dataset) for dataset in self.datasets)
        return total_length


    def __getitem__(self, idx):
        """
        Retrieves a sample by index for training and validation. Data loader.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - sample: Data sample at the specified index.
        """
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
    """
    Represents a single CSV file dataset for training the morphing model.

    This class manages loading, processing, and alignment of pose data from a CSV file and
    its corresponding video file. It supports operations like train/test splitting, keypoint alignment,
    and Procrustes alignment for pose data.
    """


    def __init__(self, csv_file_path, model_type, init=True):
        """
        Initializes the dataset with the specified CSV file and model type.

        Parameters:
        - csv_file_path (str): Path to the CSV file.
        - model_type (str): Model type ('mediapipe' or 'openpose').
        - init (bool): Whether to load data immediately.
        """
        self.csv_file_path = csv_file_path
        self.model_type = model_type
        self.csv_data = None
        self.pose_inf = None
        self.confidences_inf = None
        self.selected_columns = None
        self.par = None

        # Add any additional initialization for your SingleCSVFileDataset
        if init:
            self.csv_data, self.train_frames, self.test_frames = self.load_csv_data()
            self.pose_inf, self.confidences_inf = self.load_video_data()
            assert len(self.csv_data) == len(self.pose_inf)


    def get_camera(self):
        """
        Extracts the camera index from the CSV file path.

        Returns:
        - (int): The camera index extracted from the file path.
        """
        return int(self.csv_file_path[-5])


    def _get_csv_data(self, csv_data : pd.DataFrame):
        """
        Aligns and processes columns of the CSV data based on the selected model type.

        Parameters:
        - csv_data (pd.DataFrame): The raw CSV data to be processed.

        Returns:
        - (ndarray): The aligned and formatted CSV data as a NumPy array.
        """
        csv_data = self.align_keypoints(csv_data)
        csv_data = np.array(list(csv_data.values()))
        csv_data = csv_data.transpose((1, 0, 2))

        return csv_data


    def procrustes(self, pred, gt):
        """
        Performs Procrustes alignment to minimize the differences between predicted and ground truth poses.

        Parameters:
        - pred (ndarray): Predicted keypoint coordinates (shape: [joints, 3]).
        - gt (ndarray): Ground truth keypoint coordinates (shape: [joints, 3]).

        Returns:
        - gt (ndarray): Ground truth data, unchanged.
        - pred_hat (ndarray): Aligned predicted data after Procrustes analysis.
        """
        joint_number = gt.shape[0]
        if np.sum(np.abs(pred)) != 0:
            transposed = False
            if pred.shape[0] != 3 and pred.shape[0] != 2:
                pred = pred.T
                gt = gt.T
                transposed = True
            assert (gt.shape[1] == pred.shape[1]), "The number of joints must match."

            try:
                muX = np.mean(pred, axis=1, keepdims=True)
                muY = np.mean(gt, axis=1, keepdims=True)

                X0 = pred - muX
                Y0 = gt - muY

                var1 = np.sum(X0 ** 2)
                K = X0.dot(Y0.T)
                U, s, Vh = np.linalg.svd(K)
                V = Vh.T
                Z = np.eye(U.shape[0])
                Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
                R = V.dot(Z.dot(U.T))
                scale = np.trace(R.dot(K)) / var1
                t = muY - scale * (R.dot(muX))
                pred_hat = scale * R.dot(pred) + t
                if transposed:
                    pred_hat = pred_hat.T
            except np.linalg.LinAlgError:
                pred_hat = np.tile(np.mean(gt, axis=0), (joint_number, 1))
                R = np.identity(3)
        else:
            pred_hat = np.tile(np.mean(gt, axis=0), (joint_number, 1))
            R = np.identity(3)

        return gt, pred_hat


    def align(self, pose_inf, pose_gt):
        """
        Aligns the predicted pose data to the ground truth using Procrustes analysis.

        Parameters:
        - pose_inf (ndarray): Predicted pose data (shape: [frames, joints, 3]).
        - pose_gt (ndarray): Ground truth pose data (shape: [frames, joints, 3]).

        Returns:
        - aligned_data (ndarray): The aligned pose data.
        """
        try:
            aligned_data = []
            for x in range(pose_inf.shape[0]):
                # Loops each camera in the batch (6 total)
                # Aligns the data using Procrustes
                # We do not scale, since we already scaled and this information is stored in the Normalize class
                _, Z = self.procrustes(pose_inf[x], pose_gt)  # data[0] is the VizLab data/reference data/ground truth
                aligned_data.append(Z)
            return np.stack(aligned_data)
        except Exception as e:
            print(f"Error in normalize_and_align: {e}")
            raise e


    def get_dataset(self, train=True):
        """
        Creates a dataset object for training or testing.

        Parameters:
        - train (bool): If True, creates a training dataset; otherwise, creates a testing dataset.

        Returns:
        - dataset (SingleCSVFileDataset): A new dataset object with the corresponding data split.
        """
        dataset = SingleCSVFileDataset(self.csv_file_path, self.model_type, init=False)
        csv_data, pose_inf, confidences_inf = self.get_training_data() if train else self.get_test_data()
        # Align data according to procrustes
        dataset.csv_data = csv_data
        dataset.pose_inf = pose_inf
        dataset.confidences_inf = confidences_inf
        return dataset


    def align_procrustes(self):
        """
        Aligns predicted pose data (`pose_inf`) to the ground truth (`csv_data`) using Procrustes analysis on each frame.
        """
        for i in range(self.pose_inf.shape[0]):
            self.pose_inf[i] = self.align(self.pose_inf[i], self.csv_data[i])


    def get_training_data(self):
        return self.csv_data[self.train_frames], self.pose_inf[self.train_frames], self.confidences_inf[self.train_frames]


    def get_eval_data(self):
        return self.csv_data[self.eval_frames], self.pose_inf[self.eval_frames], self.confidences_inf[self.eval_frames]


    def get_test_data(self):
        return self.csv_data[self.test_frames], self.pose_inf[self.test_frames], self.confidences_inf[self.test_frames]


    def get_all_datasets(self):
        """
        Retrieves training, evaluation, and testing datasets as separate `SingleCSVFileDataset` objects.

        Returns:
        - training (SingleCSVFileDataset): Dataset object containing training data.
        - eval (SingleCSVFileDataset): Dataset object containing evaluation data.
        - test (SingleCSVFileDataset): Dataset object containing testing data.
        """
        training = self.get_dataset(train=True)
        eval = self.get_dataset(train=False)
        test = self.get_dataset(train=False)
        return training, eval, test


    def get_split_indexes(self, csv_data, split=[0.8, 0.2]):
        """
        Splits the data into training and testing sets based on a specified ratio. This method uses the `Iteration`
        column in the input `csv_data` to group rows into repetitions of movements. The repetitions are then split into
        training and testing subsets according to the specified ratio.

        Parameters:
        - csv_data (pd.DataFrame): The input CSV data containing at least an `Iteration` column.
        - split (list): A list specifying the split ratio for training and testing, e.g., [0.8, 0.2].

        Returns:
        - train_index (pd.Index): Indexes of rows corresponding to the training subset.
        - test_index (pd.Index): Indexes of rows corresponding to the testing subset.
        """
        # Drop NaN values in Iteration column
        iteration_values = csv_data['Iteration'].dropna().unique()

        # Perform split
        split_index = int(len(iteration_values) * split[0])

        # If train take 80% of the data, else take 20% (Must find better split)
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
        """
        Loads and processes the CSV data for the dataset. This method reads the CSV file, splits the data into training
        and testing subsets using `get_split_indexes`, and processes the data by dropping irrelevant columns and
        aligning the keypoints.

        Returns:
        - csv_data (ndarray): The aligned and processed ground truth pose data as a NumPy array.
        - train_index (pd.Index): Indexes of rows corresponding to the training subset.
        - test_index (pd.Index): Indexes of rows corresponding to the testing subset.
        """
        data = []
        train_index = None
        test_index = None
        path = self.csv_file_path
        csv_data = pd.read_csv(path)
        print(path)

        if train_index is None:
            train_index, test_index = self.get_split_indexes(csv_data)

        # Drop irrelevant columns
        csv_data.drop(columns=['Time', 'CameraFrame', 'Iteration'], inplace=True)

        # Align the columns (you may need to modify this part based on your needs)
        csv_data = self._get_csv_data(csv_data)

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

        # Define different order arrays based on alignment_identifier, specific for our ground truth dataset!
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

       ###################################################################################################
       ## This need to be adapted if trained on another model. Maybe move that to a yaml in the future! ##
       ###################################################################################################

        elif self.model_type == 'openpose':
            openpose_keypoints = {
                "Nose": "RBAK",
                "Neck": "CLAV",
                "RShoulder": "RSHO",
                "RElbow": "RELB",
                "RWrist": "RWRA",
                "LShoulder": "LSHO",
                "LElbow": "LELB",
                "LWrist": "LWRA",
                "RHip": "RASI",
                "RKnee": "RKNE",
                "RAnkle": "RANK",
                "LHip": "LASI",
                "LKnee": "LKNE",
                "LAnkle": "LANK",
                "REye": "REJC",
                "LEye": "LEJC",
                "REar": "RHEE",
                "LEar": "LHEE",
                "Background": "STRN",
                "RBigToe": "RTOE",
                "RSmallToe": "RTOE",
                "RHeel": "RHEE",
                "LBigToe": "LTOE",
                "LSmallToe": "LTOE",
                "LHeel": "LHEE"
            }
            indexes = []
            for i in openpose_keypoints.values():
                indexes.append(keypoints_org_names.index(i))

            self.selected_columns = indexes
        else:
            raise ValueError(f"Invalid model_name: {self.model_type }")

        # Extract original keypoint names
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
        """
        Loads video data and extracts pose keypoints and confidence scores. This method processes video files
        corresponding to the CSV data to extract predicted 3D pose keypoints and their associated confidence scores for
        multiple cameras. It can support multiple model types for pose estimation.

        Returns:
        - pose_keypoints (ndarray): Extracted 3D pose keypoints, with shape 
          `[frames, cameras, joints, 3]`.
        - confidences (ndarray): Confidence scores for the extracted keypoints, with 
          shape `[frames, cameras, joints]`.
        """

        data_key = []
        data_conf = []
        avi_file_path = self.csv_file_path.replace('.csv', '.avi')

        # Iterate through all 6 camera angles of the dataset
        for i in range(6):
            avi_file_path = avi_file_path.replace(f"Cam{(i-1) if i > 0 else 0}", f"Cam{i}")
            print('Start loading video data' + avi_file_path + '...')
            cap = cv2.VideoCapture(avi_file_path)

            if self.model_type == 'mediapipe':
                pose_keypoints, _, _, confidences = MediaPipe.inference_video(cap)
                confidences = confidences[:, self.selected_columns]
                print("Confidences:", confidences.shape)
                pose_keypoints = pose_keypoints[:, self.selected_columns, 1:]

            # Example for adding other models with different outputs and keypoint orders
            elif self.model_type == 'openpose':
                #pose_keypoints = OpenPose.process_video_openpose(cap)
                confidences = confidences[:, self.selected_columns, 0]
                pose_keypoints = pose_keypoints[:, self.selected_columns, 0:]

            else:
                raise ValueError(f"Invalid model_name: {self.model_type}")
            print('Finished loading video data' + avi_file_path + '...')

            data_key.append(pose_keypoints)
            data_conf.append(confidences)

        for i in data_key:
            print(i.shape)

        pose_keypoints = np.array(data_key)
        confidences = np.array(data_conf)
        pose_keypoints = np.swapaxes(pose_keypoints, 0, 1)
        confidences = np.swapaxes(confidences, 0, 1)
        return pose_keypoints, confidences

    def __len__(self):
        return len(self.csv_data)

    def filter_data(self, threshold=0.7):
        """
         Filters out low-confidence data samples based on a specified threshold.

         Parameters:
         - threshold (float): Confidence score threshold. Any sample with a keypoint confidence
         mean below this value will be excluded. Default is 0.7.
        """
        # Implement your filtering logic here
        ids = []
        for idx in range(self.__len__()):
            if np.any(np.mean(self.__getitem__(idx)['confidences_inf'], axis=0) < threshold):
                ids.append(idx)
        self.csv_data = np.array([item for idx, item in enumerate(self.csv_data) if idx not in ids])
        self.pose_inf = np.array([item for idx, item in enumerate(self.pose_inf) if idx not in ids])
        self.confidences_inf =  np.array([item for idx, item in enumerate(self.confidences_inf) if idx not in ids])

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
                         'confidences_inf': self.confidences_inf[idx], "par": self.par}

        return combined_data

    def __str__(self):
        return f"SingleCSVFileDataset({self.csv_file_path}), len={len(self)}"


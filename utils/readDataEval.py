import pandas as pd
import numpy as np
import os
import cv2


def align_keypoints(keypoints_org, model_type):
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

    if model_type == 'mediapipe':
        column_mapping = {
            'RShoulder': 'RSJC',  # 12 - 0
            'LShoulder': 'LSJC',  # 11 - 1
            'RElbow': 'REJC',  # 14 - 2
            'LElbow': 'LEJC',  # 13 - 3
            'RWrist': 'RWJC',  # 16 - 4
            'LWrist': 'LWJC',  # 15 - 5
            'RHip': 'RHJC',  # 24 - 6
            'LHip': 'LHJC',  # 23 - 7
            'RKnee': 'RKJC',  # 26 - 8
            'LKnee': 'LKJC',  # 25  - 9
            'RAnkle': 'RAJC',  # 28 - 10
            'LAnkle': 'LAJC',  # 27 - 11
            'RHeel': 'RHEE',  # 30 - 12
            'LHeel': 'LHEE',  # 29 - 13
            'RFootIndex': 'RTOE',  # 32 - 14
            'LFootIndex': 'LTOE',  # 31 - 15
        }
        selected_columns = [12, 11, 14, 13, 16, 15, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31]

    elif model_type == 'openpose':
        # More a placeholder than anything else
        column_mapping = [1, 3, 2]
        selected_columns = [12, 11, 14, 13, 16, 15, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31]
    else:
        raise ValueError(f"Invalid model_name: {model_type}")

    # Extract original keypoint names (assuming they follow a pattern)
    keypoints_org_names = set(col.rsplit('_', 1)[0] for col in keypoints_org.columns)

    # Create a dictionary with sublists (X, Y, Z) of the original keypoints
    keypoints_org_subarrays = {}

    # Group X, Y, Z coordinates for each joint
    for keypoint in keypoints_org_names:
        keypoint_cols = [f"{keypoint}_{coord}" for coord in ['X', 'Y', 'Z']]
        keypoints_org_subarrays[keypoint] = keypoints_org[keypoint_cols].to_numpy()

    # Filter columns based on column_mapping and reorder the DataFrame
    ordered_keypoints_org_subarrays = dict(
        zip(column_mapping.keys(), [keypoints_org_subarrays[key] for key in column_mapping.values()]))

    return ordered_keypoints_org_subarrays


def load_csv(self, csv_file_path):
    """
    Load the csv file from the specified path
    """
    csv_data = pd.read_csv(csv_file_path)
    # Drop irrelevant columns
    csv_data.drop(columns=['Time', 'CameraFrame', 'Iteration'], inplace=True)

    # Align the columns (you may need to modify this part based on your needs)
    csv_data = align_keypoints(csv_data)
    csv_data = np.array(list(csv_data.values()))
    csv_data = csv_data.transpose((1, 0, 2))
    return csv_data


def load_data(path, par, mov, cams):
    participant_folder = os.path.join(path, par)
    video_caps = []
    gt_keypoints = []

    def load_cam(cam):
        cam_str = f'Cam{cam}'
        mov_str = f'Mov{mov}'
        csv_path = os.path.join(participant_folder, f'{par}_{mov_str}_{cam_str}.csv')
        video_path = os.path.join(participant_folder, f'{par}_{mov_str}_{cam_str}.avi')
        print(f'Loading {csv_path}')
        points = load_csv(csv_path)
        cap = cv2.VideoCapture(video_path)
        return cam, cap, points

    if isinstance(cams, list):
        for cam in cams:
            cam_num, cap, points = load_cam(cam)
            video_caps.append((cam_num, cap))
            gt_keypoints.append((cam_num, points))

    elif isinstance(cams, int):
        cam_num, cap, points = load_cam(cams)
        video_caps.append((cam_num, cap))
        gt_keypoints.append((cam_num, points))

    else:
        raise ValueError(f'Invalid type for cams: {type(cams)}')

    return gt_keypoints, video_caps
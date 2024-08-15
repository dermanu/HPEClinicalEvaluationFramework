import pandas as pd
import numpy as np
import os
import cv2


def align_keypoints(keypoints_org):
    """
    Aligns keypoints in a DataFrame based on different alignment identifiers.
    :param keypoints_org (pd.DataFrame or None): Original DataFrame containing the keypoints.
    :param model_type (str): String identifier for selecting the alignment array based on the respective model.
    :return aligned_keypoints (dict): Dictionary containing aligned keypoints.
    """
    # Only get relevant datapoints for analysis and give them better names for debugging
    mapping = {
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

    # Extract original keypoint names (assuming they follow a pattern)
    keypoints_org_names = set(col.rsplit('_', 1)[0] for col in keypoints_org.columns)

    # Create a dictionary with sublists (X, Y, Z) of the original keypoints
    keypoints_org_subarrays = {}

    # Group X, Y, Z coordinates for each joint
    for keypoint in keypoints_org_names:
        keypoint_cols = [f"{keypoint}_{coord}" for coord in ['X', 'Y', 'Z']]
        keypoints_org_subarrays[keypoint] = keypoints_org[keypoint_cols].to_numpy()

    # Filter columns based on column_mapping and reorder the DataFrame
    ordered_keypoints = {key: keypoints_org_subarrays[val] for key, val in mapping.items() if val in keypoints_org_subarrays}

    return ordered_keypoints


def load_csv(csv_file_path):
    """
    Load keypoints from a CSV file.
    :param csv_file_path (str): Path to the CSV file.
    :param model_type (str): String identifier for the model used.
    :return keypoints (np.array): Array containing keypoints.
    """
    # Load the CSV file
    keypoints_org = pd.read_csv(csv_file_path)
    # Drop irrelevant columns
    keypoints_org.drop(columns=['Time', 'CameraFrame', 'Iteration'], inplace=True)
    # Align the columns
    aligned_keypoints = align_keypoints(keypoints_org)
    keypoints = np.array(list(aligned_keypoints.values())).transpose((1, 0, 2))
    return keypoints


def load_data(path, par, mov, cams):
    """
    Load data for a participant.
    :param path (str): Root directory containing participant data.
    :param par (str): Participant identifier.
    :param mov (str): Movement identifier.
    :param cams (int or list): Camera(s) to load data from.
    :param model_type (str): String identifier for the model used.
    :return gt_keypoints (list): List of tuples containing camera number and keypoints.
    :return video_caps (list): List of tuples containing camera number and video capture object.
    """
    participant_folder = os.path.join(path, par)
    gt_keypoints = []
    video_caps = []

    def load_cam(cam):
        cam_str = f'Cam{cam}'
        mov_str = f'Mov{mov}'
        csv_path = os.path.join(participant_folder, f'{par}_{mov_str}_{cam_str}.csv')
        video_path = os.path.join(participant_folder, f'{par}_{mov_str}_{cam_str}.avi')

        if not os.path.exists(csv_path) or not os.path.exists(video_path):
            print(f"Warning: CSV or video file not found for camera {cam}.")
            return None, None, None

        keypoints = load_csv(csv_path)
        cap = cv2.VideoCapture(video_path)
        return cam, keypoints, cap

    # Load data for each camera
    if isinstance(cams, list):
        for cam in cams:
            cam_num, keypoints, cap = load_cam(cam)
            if cam_num is not None:
                gt_keypoints.append((cam_num, keypoints))
                video_caps.append((cam_num, cap))
    elif isinstance(cams, int):
        cam_num, keypoints, cap = load_cam(cams)
        if cam_num is not None:
            gt_keypoints.append((cam_num, keypoints))
            video_caps.append((cam_num, cap))
    else:
        raise ValueError(f'Invalid type for cams: {type(cams)}')

    return gt_keypoints, video_caps
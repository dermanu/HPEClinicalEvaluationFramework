from mpi4py import MPI
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
import yaml
import time
from models.dlt import DLT, weighted_DLT
from utils.frameAugmentation import FrameAugmentor
import os
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

num_cameras = size - 1

if num_cameras < 1:
    if rank == 0:
        print("Please run with at least one worker (camera).")
    MPI.Finalize()
    exit()

# Shared configurations
model_path = 'pose_landmarker_full.task'  # Adjust as needed
sweep_config = None  # Replace with actual configuration if necessary

def load_projection_matrices(file_path='P_values.yaml'):
    with open(file_path, 'r') as yaml_file:
        P_dict = yaml.safe_load(yaml_file)
    projection_matrices = {}
    for key, value in P_dict.items():
        projection_matrices[key] = np.array(value)
    return projection_matrices

def write_keypoints_to_disk(filename, kpts):
    with open(filename, 'w') as fout:
        for frame_kpts in kpts:
            for kpt in frame_kpts:
                fout.write(' '.join(map(str, kpt)) + ' ')
            fout.write('\n')

def init_worker():
    global pose_landmarker
    global frame_augmentor

    pose_landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE
        )
    )

    if sweep_config and sweep_config.get('augmentation', 'none') != "none":
        frame_augmentor = FrameAugmentor()
    else:
        frame_augmentor = None

def process_frame(frame):
    global pose_landmarker
    global frame_augmentor

    if frame is None:
        return None, None

    # Convert and rotate frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.rotate(rgb_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Apply frame augmentation if available
    if frame_augmentor is not None:
        rgb_frame = frame_augmentor.augment_frames(rgb_frame)

    height, width = rgb_frame.shape[:2]

    # Detect pose landmarks
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = pose_landmarker.detect(mp_image)
    landmarks = results.pose_landmarks if results.pose_landmarks else None

    # Collect keypoints and confidences
    num_landmarks = 33
    keypoints = np.full((num_landmarks, 2), np.nan)
    confidences = np.full(num_landmarks, np.nan)

    if landmarks:
        for i, landmark in enumerate(landmarks[0]):
            pxl_x = landmark.x * width
            pxl_y = landmark.y * height
            keypoints[i, 0] = int(round(pxl_x))
            keypoints[i, 1] = int(round(pxl_y))
            confidences[i] = landmark.visibility

    return keypoints, confidences

def main_master():
    projection_matrices = load_projection_matrices()

    keypoints_data = []
    frame_number = 0

    while True:
        frame_keypoints = np.full((33, num_cameras, 2), np.nan)
        confidences = np.full((33, num_cameras), np.nan)
        end_of_stream = False

        for cam_idx in range(1, num_cameras + 1):
            data = comm.recv(source=cam_idx, tag=frame_number)
            if data is None:
                end_of_stream = True
                break
            keypoints, conf = data
            frame_keypoints[:, cam_idx - 1, :] = keypoints
            confidences[:, cam_idx - 1] = conf

        if end_of_stream:
            print("All camera streams have ended.")
            break

        # Triangulate 3D points
        points_3d = np.full((33, 3), np.nan)
        for landmark_idx in range(33):
            valid_indices = ~np.isnan(frame_keypoints[landmark_idx, :, 0])
            num_valid = np.sum(valid_indices)
            if num_valid >= 2:
                keypoints_2d = frame_keypoints[landmark_idx, valid_indices, :]
                cams = np.where(valid_indices)[0]
                projection_keys = [f"Camera_{cam}" for cam in cams]
                projections_valid = [projection_matrices[key] for key in projection_keys]
                try:
                    # Adjust DLT function as per your implementation
                    point_3d = DLT(projections_valid, keypoints_2d[np.newaxis, :, :])
                    points_3d[landmark_idx] = point_3d
                except Exception as e:
                    print(f"Triangulation error for landmark {landmark_idx} in frame {frame_number}: {e}")
            else:
                print(f"Not enough data to triangulate landmark {landmark_idx} in frame {frame_number}.")

        keypoints_data.append(points_3d)
        frame_number += 1

    # Optionally, send termination signal to workers
    for cam_idx in range(1, num_cameras + 1):
        comm.send(None, dest=cam_idx, tag=0)

    # Save results
    keypoints_data = np.array(keypoints_data)
    write_keypoints_to_disk('kpts_3D.dat', keypoints_data)
    print("Triangulation complete and results saved.")

def main_worker(cam_index, video_path):
    # Initialize PoseLandmarker and FrameAugmentor
    init_worker()

    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, conf = process_frame(frame)

        # Send keypoints to master
        comm.send((keypoints, conf), dest=0, tag=frame_number)
        frame_number += 1

    cap.release()
    print(f"Camera {cam_index} processing complete.")

if rank == 0:
    main_master()
else:
    # Assign camera indices based on rank
    cam_idx = rank - 1

    # Define video paths for each camera
    video_paths = {
        0: '/home/emanu/Desktop/MoCap/segmented/par10/par10_Mov14_Cam0.avi',
        1: '/home/emanu/Desktop/MoCap/segmented/par10/par10_Mov14_Cam4.avi',
        # Add more cameras as needed
    }

    if cam_idx not in video_paths:
        print(f"No video path assigned for camera index {cam_idx}.")
        comm.send(None, dest=0, tag=0)
    else:
        video_path = video_paths[cam_idx]
        main_worker(cam_idx, video_path)

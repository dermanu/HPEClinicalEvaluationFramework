import cv2
import numpy as np
import time
from utils.frameAugmentation import FrameAugmentor
import mediapipe as mp
from mediapipe.tasks.python import vision
from models.dlt import triangulate_from_multiple_views_svd, triangulate_from_multiple_views_sii

# Load the pose landmarker model once to avoid reloading it multiple times
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='models/pose_landmarker_heavy.task'),
    running_mode=mp.tasks.vision.RunningMode.IMAGE)
PoseLandmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)


def inference_video(caps, projections, sweep_config=None):
    if sweep_config is not None:
        # Initialize frame augmentor
        frameaug = FrameAugmentor()

    keypoints_data = []
    inference_time = []
    frame_number = 0
    num_cameras = len(caps)

    while all(cap[1].isOpened() for cap in caps):
        frames = []
        for cap in caps:
            ret, frame = cap[1].read()
            if not ret:
                break
            frames.append(frame)

        if len(frames) != num_cameras:
            break

        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

        if sweep_config is not None:
            # Augment frames
            rgb_frames = [frameaug.augment_frames(rgb_frame, sweep_config) for rgb_frame in rgb_frames]

        frame_keypoints = np.zeros((33, num_cameras, 2))
        # Record start time
        start_time = time.time()

        for cam_idx, rgb_frame in enumerate(rgb_frames):
            # Convert the frame to a MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect pose landmarks from the input image
            results = PoseLandmarker.detect(mp_image)

            # If pose detected, save keypoints to data list
            if results.pose_world_landmarks:
                landmarks = results.pose_world_landmarks
                for idx in range(len(landmarks)):
                    pose_landmarks = landmarks[idx]
                    frame_keypoints[:, cam_idx, 0] = np.array([landmark.x for i, landmark in enumerate(pose_landmarks)])
                    frame_keypoints[:, cam_idx, 1] = np.array([landmark.y for i, landmark in enumerate(pose_landmarks)])

        frame_number += 1

        # Perform multiocular estimation for the current frame
        points_3d = triangulate_from_multiple_views_sii(projections, frame_keypoints)

        # Record end time
        end_time = time.time()
        inference_time.append(end_time - start_time)

        # Store the 3D points for the current frame
        keypoints_data.append(points_3d)

    # Release resources
    for cap in caps:
        cap.release()

    # Convert inference time to numpy array
    inference_time = np.array(inference_time)

    # Return keypoints data to NumPy array, save last frame for logging
    return np.array(keypoints_data), inference_time, rgb_frames[0]


# Example usage
#caps = [cv2.VideoCapture(0), cv2.VideoCapture(1)]  # Add more cameras as needed
#projections = [...]  # Your camera projection matrices
#sweep_config = ...  # Your sweep configuration if any

#points_3d, inference_time, last_frame = inference_video(caps, projections, sweep_config)

# Process points_3d, inference_time, and last_frame as needed

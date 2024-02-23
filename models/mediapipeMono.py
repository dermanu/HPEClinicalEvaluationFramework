import cv2
import mediapipe as mp
import numpy as np
import time
from utils.frameAugmentation import FrameAugmentor


# Currently runs on CPU as per default.
def inference_video(cap, sweep_config=None, mp_complexity=2, dimensions=3):
    if sweep_config is not None:
        # Initialize frame augmentor
        frameaug = FrameAugmentor()

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    # Model complexity set to 2 for mono-ocular. Minimum detection and tracking confidence set to 0.5 as default.
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=mp_complexity, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    keypoints_data = []
    inference_time = []

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if sweep_config is not None:
            # Augment frame
            frame = frameaug.augment_frames(frame, sweep_config)

        # Record start time
        start_time = time.time()

        # Convert the BGR image to RGB - Is this really needed?
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = pose.process(rgb_frame)

        # Record end time
        end_time = time.time()

        inference_time.append(end_time - start_time)

        # If pose detected, save keypoints to data list
        if results.pose_landmarks:
            # Save array dependent on wanted dimensions
            if dimensions == 3:
                frame_data = np.array([[i, landmark.x, landmark.y, landmark.z] for i, landmark in
                                       enumerate(results.pose_landmarks.landmark)])
            elif dimensions == 2:
                frame_data = np.array([[i, landmark.x, landmark.y] for i, landmark in
                                       enumerate(results.pose_landmarks.landmark)])
            keypoints_data.append(frame_data)

        frame_number += 1

    # Release resources
    cap.release()

    # Return keypoints data to NumPy array, save last frame for logging
    return np.array(keypoints_data), np.array(inference_time), rgb_frame

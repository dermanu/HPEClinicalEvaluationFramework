import cv2
import numpy as np
import time
from utils.frameAugmentation import FrameAugmentor
import mediapipe as mp
from mediapipe.tasks.python import vision

# Load the pose landmarker model once to avoid reloading it multiple times
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='..\models\pose_landmarker_heavy.task'),
    running_mode=mp.tasks.vision.RunningMode.IMAGE)
PoseLandmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

# Currently runs on CPU as per default.
def inference_video(cap, sweep_config=None, dimensions=3):
    if sweep_config is not None:
        # Initialize frame augmentor
        frameaug = FrameAugmentor()

    keypoints_data = []
    confidence_data = []  # List to store confidence scores
    inference_time = []
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB - Is this really needed?
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if sweep_config is not None:
            # Augment frame
            rgb_frame = frameaug.augment_frames(rgb_frame, sweep_config)

        # Record start time
        start_time = time.time()

        # Convert the frame to a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect pose landmarks from the input image
        results = PoseLandmarker.detect(mp_image)

        # Record end time
        end_time = time.time()

        inference_time.append(end_time - start_time)

        # If pose detected, save keypoints to data list
        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks

            for idx in range(len(landmarks)):
                pose_landmarks = landmarks[idx]

                # Save array dependent on wanted dimensions
                if dimensions == 3:
                    frame_data = np.array([[i, landmark.x, landmark.y, landmark.z]
                                           for i, landmark in enumerate(pose_landmarks)])
                elif dimensions == 2:
                    frame_data = np.array([[i, landmark.x, landmark.y] for i, landmark in enumerate(pose_landmarks)])

                confidences = np.array([landmark.visibility for landmark in pose_landmarks])

            #print(np.mean(np.array(confidences), axis=0))
            confidence_data.append(confidences)
            keypoints_data.append(frame_data)

        frame_number += 1

    # Release resources
    cap.release()

    # Return keypoints data to NumPy array, save last frame for logging
    return np.array(keypoints_data), np.array(inference_time), rgb_frame, np.array(confidence_data)

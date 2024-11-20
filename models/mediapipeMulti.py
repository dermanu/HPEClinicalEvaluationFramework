import cv2
cv2.setUseOptimized(True)
cv2.setNumThreads(2)
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.frameAugmentation import FrameAugmentor
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from models.dlt import DLT, weighted_DLT

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Load the pose landmarker model once to avoid reloading it multiple times
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/pose_landmarker_heavy.task'),
    running_mode=VisionRunningMode.IMAGE)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)


def process_frame(cap, frameaug=None, sweep_config=None):
    ret, frame = cap.read()
    # print(f"Frame read status: {ret}")  # Debug: Check if frame is read
    if not ret:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.rotate(rgb_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if sweep_config._items['augmentation'] != "none" and frameaug is not None:
        rgb_frame = frameaug.augment_frames(rgb_frame, sweep_config)

    return rgb_frame


def detect_pose(rgb_frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = pose_landmarker.detect(mp_image)
    if results.pose_landmarks and results.pose_landmarks[0]:
        # Return the first detected pose's landmarks
        return results.pose_landmarks[0]
    else:
        return None


def inference_video(caps, projections, sweep_config=None):
    if sweep_config._items['augmentation'] != "none":
        frameaug = FrameAugmentor()
    else:
        frameaug = None

    keypoints_data = []
    inference_time = []
    frame_number = 0
    num_cameras = len(caps)
    last_rgb_frame = None

    with ThreadPoolExecutor(max_workers=num_cameras) as executor:
        while all(cap.isOpened() for _, cap in caps):
            # Process frames in parallel
            futures = {executor.submit(process_frame, cap[1], frameaug, sweep_config): cap[0] for cap in caps}
            rgb_frames = [None] * num_cameras
            for future in as_completed(futures):
                cam_idx = futures[future]
                try:
                    rgb_frame = future.result()
                    if rgb_frame is not None:
                        rgb_frames[cam_idx] = rgb_frame
                except Exception as e:
                    print(f"Error processing frame: {e}")

            # Skip frame if no frames are read
            if all(frame is None for frame in rgb_frames):
                print(f"No frames read for frame {frame_number}, skipping.")
                frame_number += 1
                continue

            frame_keypoints = np.full((33, num_cameras, 2), np.nan)  # Initialize with NaNs
            confidences = np.full((33, num_cameras), np.nan)          # Initialize with NaNs

            start_time = time.time()
            # Detect poses in parallel
            pose_futures = {executor.submit(detect_pose, rgb_frames[cam_idx]): cam_idx for cam_idx in
                            range(num_cameras) if rgb_frames[cam_idx] is not None}

            for future in as_completed(pose_futures):
                cam_idx = pose_futures[future]
                landmarks = future.result()
                if landmarks:
                    frame_keypoints[:, cam_idx, 0] = [landmark.x for landmark in landmarks]
                    frame_keypoints[:, cam_idx, 1] = [landmark.y for landmark in landmarks]
                    confidences[:, cam_idx] = [landmark.visibility for landmark in landmarks]
                else:
                    print(f"No valid detection from camera {cam_idx}.")

            # Proceed even if some cameras didn't detect landmarks
            points_3d = np.full((33, 3), np.nan)  # Initialize 3D points with NaNs

            for landmark_idx in range(33):
                # Collect data for the current landmark across cameras
                valid_cameras = ~np.isnan(frame_keypoints[landmark_idx, :, 0])
                num_valid_cameras = np.sum(valid_cameras)

                if num_valid_cameras >= 2:
                    # Collect 2D keypoints and projection matrices for valid cameras
                    keypoints_2d = frame_keypoints[landmark_idx, valid_cameras, :]
                    projections_valid = [projections[i] for i in np.where(valid_cameras)[0]]

                    # Perform triangulation for the current landmark
                    point_3d = DLT(projections_valid, keypoints_2d[np.newaxis, :, :])[0]
                    points_3d[landmark_idx] = point_3d
                else:
                    # Not enough data to triangulate this landmark
                    print(f"Not enough data to triangulate landmark {landmark_idx} in frame {frame_number}.")

            keypoints_data.append(points_3d)
            end_time = time.time()
            inference_time.append(end_time - start_time)
            frame_number += 1

            # Adjust rotation if needed
            rgb_frames = [np.rot90(frame, k=-1) if frame is not None else None for frame in rgb_frames]
            # Concatenate frames that are not None
            valid_frames = [frame for frame in rgb_frames if frame is not None]
            if valid_frames:
                last_rgb_frame = np.concatenate(valid_frames, axis=1)

    for _, cap in caps:
        cap.release()

    inference_time = np.array(inference_time)
    return np.array(keypoints_data), inference_time, last_rgb_frame

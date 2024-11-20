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
    if frameaug is not None and sweep_config is not None:
        rgb_frame = frameaug.augment_frames(rgb_frame, sweep_config)

    return rgb_frame


def detect_pose(rgb_frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = pose_landmarker.detect(mp_image)
    # print(f"Landmarks detected: {results.pose_landmarks is not None}")  # Debug: Check detection
    return results.pose_landmarks if results.pose_landmarks else None


def inference_video(caps, projections, sweep_config=None):
    frameaug = FrameAugmentor() if sweep_config is not None else None

    keypoints_data = []
    inference_time = []
    frame_number = 0
    num_cameras = len(caps)
    last_rgb_frame = None

    with ThreadPoolExecutor(max_workers=num_cameras) as executor:
        while all(cap.isOpened() for _, cap in caps):
            futures = [executor.submit(process_frame, cap[1], frameaug, sweep_config) for cap in caps]
            rgb_frames = [None] * num_cameras
            for future in as_completed(futures):
                cam_idx = futures[future]
                try:
                    rgb_frame = future.result()
                    if rgb_frame is not None:
                        rgb_frames[cam_idx] = rgb_frame
                except Exception as e:
                    print(f"Error processing frame: {e}")

            if None in rgb_frames:
                print(f"Incomplete frames for frame {frame_number}, skipping.")
                continue

            frame_keypoints = np.zeros((33, num_cameras, 2))
            confidences = np.zeros((33, num_cameras))
            valid_detections = True  # Flag to check if all cameras detected poses

            start_time = time.time()
            pose_futures = {executor.submit(detect_pose, rgb_frames[cam_idx]): cam_idx for cam_idx in
                            range(num_cameras)}

            for future in as_completed(pose_futures):
                cam_idx = pose_futures[future]
                landmarks = future.result()
                if landmarks:
                    frame_keypoints[:, cam_idx, 0] = [landmark.x for landmark in landmarks]
                    frame_keypoints[:, cam_idx, 1] = [landmark.y for landmark in landmarks]
                    confidences[:, cam_idx] = [landmark.visibility for landmark in landmarks]
                else:
                    valid_detections = False
                    print(f"No valid detection from camera {cam_idx}. Skipping triangulation.")
                    break  # Exit the loop if any detection is invalid

            if not valid_detections:
                keypoints_data.append(np.full((33, 3), np.nan))
                inference_time.append(np.nan)
                frame_number += 1
                print(f"Skipping frame {frame_number} due to invalid detections.")
                continue

            # Triangulate 3D points
            points_3d = DLT(projections, frame_keypoints)
            keypoints_data.append(points_3d)
            end_time = time.time()
            inference_time.append(end_time - start_time)
            frame_number += 1

            # Adjust rotation if needed
            rgb_frames = [np.rot90(frame, k=-1) for frame in rgb_frames]
            last_rgb_frame = np.concatenate(rgb_frames, axis=1)

    for _, cap in caps:
            cap.release()

    inference_time = np.array(inference_time)
    return np.array(keypoints_data), inference_time, last_rgb_frame
import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.frameAugmentation import FrameAugmentor
import mediapipe as mp
from mediapipe.tasks.python import vision
from models.dlt import triangulate_from_multiple_views_svd, triangulate_from_multiple_views_sii
import torch

# Load the pose landmarker model once to avoid reloading it multiple times
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='models/pose_landmarker_full.task'),
    running_mode=mp.tasks.vision.RunningMode.IMAGE)
PoseLandmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

def process_frame(cap, frameaug, sweep_config):
    ret, frame = cap.read()
    if not ret:
        return None, None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if sweep_config is not None:
        rgb_frame = frameaug.augment_frames(rgb_frame, sweep_config)

    return frame, rgb_frame

def detect_pose(rgb_frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = PoseLandmarker.detect(mp_image)
    return results.pose_world_landmarks if results.pose_world_landmarks else []

def inference_video(caps, projections, sweep_config=None):
    if sweep_config is not None:
        frameaug = FrameAugmentor()

    keypoints_data = []
    inference_time = []
    frame_number = 0
    num_cameras = len(caps)
    last_rgb_frame = None

    with ThreadPoolExecutor(max_workers=num_cameras) as executor:
        while all(cap.isOpened() for _, cap in caps):
            futures = [executor.submit(process_frame, cap[1], frameaug, sweep_config) for cap in caps]
            frames = []
            rgb_frames = []
            for future in as_completed(futures):
                try:
                    frame, rgb_frame = future.result()
                    if frame is not None:
                        frames.append(frame)
                        rgb_frames.append(rgb_frame)
                except Exception as e:
                    print(f"Error processing frame: {e}")

            if len(frames) != num_cameras:
                break

            frame_keypoints = np.zeros((33, num_cameras, 2))

            start_time = time.time()

            pose_futures = [executor.submit(detect_pose, rgb_frame) for rgb_frame in rgb_frames]
            for cam_idx, future in enumerate(as_completed(pose_futures)):
                try:
                    landmarks = future.result()
                    for idx, pose_landmarks in enumerate(landmarks):
                        frame_keypoints[:, cam_idx, 0] = [landmark.x for landmark in pose_landmarks]
                        frame_keypoints[:, cam_idx, 1] = [landmark.y for landmark in pose_landmarks]
                except Exception as e:
                    print(f"Error detecting pose: {e}")

            points_3d = triangulate_from_multiple_views_sii(projections, frame_keypoints, number_of_iterations=2)

            end_time = time.time()
            inference_time.append(end_time - start_time)

            keypoints_data.append(points_3d.cpu().numpy())
            frame_number += 1

            if len(rgb_frames) >= 2:
                last_rgb_frame = np.concatenate((rgb_frames[0], rgb_frames[1]), axis=1)
            elif len(rgb_frames) == 1:
                last_rgb_frame = rgb_frames[0]

    for _, cap in caps:
        cap.release()

    inference_time = np.array(inference_time)
    return np.array(keypoints_data), inference_time, last_rgb_frame
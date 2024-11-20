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


def process_frame(frame, frameaug=None, sweep_config=None):
    if frame is None:
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

    # Initialize cap status to keep track of which videos have ended
    cap_status = [True] * num_cameras  # True means the video is still open
    caps_dict = dict(caps)  # Convert caps to a dict for easy access

    with ThreadPoolExecutor(max_workers=num_cameras) as executor:
        while any(cap_status):  # Continue until all videos have been processed
            frames = [None] * num_cameras  # Initialize frames list

            # Read frames from caps in the main thread
            for cap_idx in range(num_cameras):
                if cap_status[cap_idx]:
                    ret, frame = caps_dict[cap_idx].read()
                    if not ret:
                        frames[cap_idx] = None
                        cap_status[cap_idx] = False  # Mark this cap as finished
                        caps_dict[cap_idx].release()
                    else:
                        frames[cap_idx] = frame
                else:
                    frames[cap_idx] = None  # No more frames from this cap

            # Process frames in parallel
            futures = {executor.submit(process_frame, frames[cap_idx], frameaug, sweep_config): cap_idx for cap_idx in range(num_cameras)}
            rgb_frames = [None] * num_cameras
            for future in as_completed(futures):
                cam_idx = futures[future]
                try:
                    rgb_frame = future.result()
                    if rgb_frame is not None:
                        rgb_frames[cam_idx] = rgb_frame
                except Exception as e:
                    print(f"Error processing frame from camera {cam_idx}: {e}")

            # Initialize keypoints and confidences with NaNs for this frame
            frame_keypoints = np.full((33, num_cameras, 2), np.nan)
            confidences = np.full((33, num_cameras), np.nan)

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
                    print(f"No valid detection from camera {cam_idx} at frame {frame_number}.")

            # Triangulate 3D points for this frame
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

        # Release any remaining video captures
        for cap_idx in range(num_cameras):
            if caps_dict[cap_idx].isOpened():
                caps_dict[cap_idx].release()

    inference_time = np.array(inference_time)
    return np.array(keypoints_data), inference_time, last_rgb_frame
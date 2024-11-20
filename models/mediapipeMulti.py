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
    if frameaug is not None and sweep_config is not None:
        rgb_frame = frameaug.augment_frames(rgb_frame, sweep_config)

    return rgb_frame


def detect_pose(rgb_frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = pose_landmarker.detect(mp_image)
    # print(f"Landmarks detected: {results.pose_landmarks is not None}")  # Debug: Check detection
    return results.pose_landmarks if results.pose_landmarks else None


def inference_video(caps, projections, sweep_config=None):
    if sweep_config and sweep_config._items.get('augmentation', 'none') != "none":
        frameaug = FrameAugmentor()
    else:
        frameaug = None

    keypoints_data = []
    inference_time = []
    frame_number = 0
    num_cameras = len(caps)
    last_rgb_frame = None

    # Extract camera numbers from caps
    cam_indices = [cam for cam, _ in caps]

    # Map camera numbers to sequential indices
    cam_to_idx = {cam: idx for idx, cam in enumerate(cam_indices)}
    idx_to_cam = {idx: cam for cam, idx in cam_to_idx.items()}

    # Initialize cap status to keep track of which videos have ended
    cap_status = {cam: True for cam in cam_indices}  # Use camera numbers as keys

    # Convert caps to a dictionary
    caps_dict = dict(caps)

    with ThreadPoolExecutor(max_workers=num_cameras) as executor:
        while any(cap_status.values()):
            frames = {}  # Use a dict to store frames by camera number

            # Read frames from caps in the main thread
            for cam in cam_indices:
                if cap_status[cam]:
                    ret, frame = caps_dict[cam].read()
                    if not ret:
                        frames[cam] = None
                        cap_status[cam] = False  # Mark this cap as finished
                        caps_dict[cam].release()
                    else:
                        frames[cam] = frame
                else:
                    frames[cam] = None  # No more frames from this cap

            # Process frames in parallel
            futures = {
                executor.submit(process_frame, frames[cam], frameaug, sweep_config): cam
                for cam in cam_indices if frames[cam] is not None
            }
            rgb_frames = {}
            for future in as_completed(futures):
                cam = futures[future]
                try:
                    rgb_frame = future.result()
                    if rgb_frame is not None:
                        rgb_frames[cam] = rgb_frame
                except Exception as e:
                    print(f"Error processing frame from camera {cam}: {e}")

            # Initialize keypoints and confidences with NaNs for this frame
            frame_keypoints = np.full((33, num_cameras, 2), np.nan)
            confidences = np.full((33, num_cameras), np.nan)

            start_time = time.time()
            # Detect poses in parallel
            pose_futures = {
                executor.submit(detect_pose, rgb_frames[cam]): cam
                for cam in rgb_frames
            }

            for future in as_completed(pose_futures):
                cam = pose_futures[future]
                idx = cam_to_idx[cam]
                landmarks = future.result()
                if landmarks:
                    frame_keypoints[:, idx, 0] = [landmark.x for landmark in landmarks[0]]
                    frame_keypoints[:, idx, 1] = [landmark.y for landmark in landmarks[0]]
                    confidences[:, idx] = [landmark.visibility for landmark in landmarks[0]]
                else:
                    print(f"No valid detection from camera {cam} at frame {frame_number}.")

            # Triangulate 3D points for this frame
            points_3d = np.full((33, 3), np.nan)  # Initialize 3D points with NaNs

            for landmark_idx in range(33):
                # Collect data for the current landmark across cameras
                valid_indices = ~np.isnan(frame_keypoints[landmark_idx, :, 0])
                num_valid_cameras = np.sum(valid_indices)

                if num_valid_cameras >= 2:
                    # Collect 2D keypoints and projection matrices for valid cameras
                    keypoints_2d = frame_keypoints[landmark_idx, valid_indices, :]
                    confidences_valid = confidences[landmark_idx, valid_indices]
                    valid_idxs = np.where(valid_indices)[0]  # Sequential indices

                    # Use sequential indices to access projections
                    projections_valid = [projections[idx] for idx in valid_idxs]

                    # Perform triangulation for the current landmark
                    point_3d = DLT(projections_valid, keypoints_2d[np.newaxis, :, :])
                    points_3d[landmark_idx] = point_3d
                else:
                    # Not enough data to triangulate this landmark
                    print(f"Not enough data to triangulate landmark {landmark_idx} in frame {frame_number}.")

            keypoints_data.append(points_3d)
            end_time = time.time()
            inference_time.append(end_time - start_time)
            frame_number += 1

            # Adjust rotation if needed
            rotated_frames = [
                np.rot90(rgb_frames[cam], k=-1) if cam in rgb_frames else None
                for cam in cam_indices
            ]
            # Concatenate frames that are not None
            valid_frames = [frame for frame in rotated_frames if frame is not None]
            if valid_frames:
                last_rgb_frame = np.concatenate(valid_frames, axis=1)

        # Release any remaining video captures
        for cam in cam_indices:
            if caps_dict[cam].isOpened():
                caps_dict[cam].release()

    inference_time = np.array(inference_time)
    return np.array(keypoints_data), inference_time, last_rgb_frame
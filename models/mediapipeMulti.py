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

    height, width = rgb_frame.shape[:2]

    return rgb_frame, (width, height)


def detect_pose(rgb_frame):
    """Detects pose landmarks in the given RGB frame."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = pose_landmarker.detect(mp_image)
    return results.pose_landmarks if results.pose_landmarks else None


def inference_video(caps, projections, sweep_config=None):
    """
    Processes video frames from multiple cameras, detects poses, scales landmarks to pixel coordinates,
    triangulates 3D points, and aggregates keypoint data.

    Parameters:
    - caps: List of tuples (camera_index, cv2.VideoCapture object)
    - projections: List of projection matrices corresponding to each camera
    - sweep_config: Configuration for frame augmentation (optional)

    Returns:
    - keypoints_data: NumPy array of triangulated 3D keypoints
    - inference_time: NumPy array of inference times per frame
    - last_rgb_frame: Concatenated RGB frames from all cameras for the last processed frame
    """
    if sweep_config and sweep_config._items.get('augmentation', 'none') != "none":
        frameaug = FrameAugmentor()
    else:
        frameaug = None

    keypoints_data = []
    inference_time = []
    frame_number = 0
    num_cameras = len(caps)
    last_rgb_frame = None

    cam_indices = [cam for cam, _ in caps]
    cam_to_idx = {cam: idx for idx, cam in enumerate(cam_indices)}
    cap_status = {cam: True for cam in cam_indices}
    caps_dict = dict(caps)


    with ThreadPoolExecutor(max_workers=num_cameras) as executor:
        while any(cap_status.values()):
            frames = {}
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

            # Skip processing if no valid frames are left
            if not any(frame is not None for frame in frames.values()):
                break

            # Process frames in parallel
            futures = {
                executor.submit(
                    lambda cam: process_frame(frames[cam], frameaug, sweep_config), cam
                ): cam for cam in cam_indices if frames[cam] is not None
            }

            rgb_frames = {}
            frame_dimensions = {}
            for future in as_completed(futures):
                cam = futures[future]
                try:
                    rgb_frame, dimensions = future.result()
                    if rgb_frame is not None:
                        rgb_frames[cam] = rgb_frame
                        frame_dimensions[cam] = dimensions
                except Exception as e:
                    print(f"Error processing frame from camera {cam}: {e}")

            # Initialize keypoints and confidences with NaNs for this frame
            frame_keypoints = np.full((33, num_cameras, 2), np.nan)
            confidences = np.full((33, num_cameras), np.nan)

            start_time = time.time()
            # Detect poses in parallel
            pose_futures = {
                executor.submit(detect_pose, rgb_frames[cam]): cam for cam in rgb_frames
            }

            for future in as_completed(pose_futures):
                cam = pose_futures[future]
                idx = cam_to_idx[cam]
                landmarks = future.result()
                if landmarks:
                    for i, landmark in enumerate(landmarks[0]):
                        pxl_x = landmark.x * frame_dimensions[cam][0]
                        pxl_y = landmark.y * frame_dimensions[cam][1]
                        pxl_x = int(round(pxl_x))
                        pxl_y = int(round(pxl_y))
                        frame_keypoints[i, idx, 0] = pxl_x
                        frame_keypoints[i, idx, 1] = pxl_y
                        confidences[i, idx] = landmark.visibility

                    #frame_keypoints[:, idx, 0] = [landmark.x for landmark in landmarks[0]]
                    #frame_keypoints[:, idx, 1] = [landmark.y for landmark in landmarks[0]]
                    #confidences[:, idx] = [landmark.visibility for landmark in landmarks[0]]
                #else:
                #    print(f"No valid detection from camera {cam} at frame {frame_number}.")

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
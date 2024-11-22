import cv2
cv2.setUseOptimized(True)
cv2.setNumThreads(2)
import numpy as np
import time
from multiprocessing import Pool
import mediapipe as mp
from mediapipe.tasks.python import vision
from models.dlt import DLT
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# Mediapipe setup
BaseOptions = mp.tasks.BaseOptions
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variables for multiprocessing workers
pose_landmarker = None
frame_augmentor = None

def init_worker(model_path, sweep_config):
    """
    Initializer for each worker process. Initializes PoseLandmarker.
    """
    global pose_landmarker

    # Initialize PoseLandmarker
    pose_landmarker = vision.PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE
        )
    )

def process_frame(args):
    """
    Processes a single frame: detects pose and extracts keypoints.
    """
    cam, frame = args
    if frame is None:
        return cam, None, None

    global pose_landmarker

    # Convert and rotate frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.rotate(rgb_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    height, width = rgb_frame.shape[:2]

    # Detect pose landmarks
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = pose_landmarker.detect(mp_image)
    landmarks = results.pose_landmarks if results.pose_landmarks else None

    # Collect keypoints and confidences
    num_landmarks = 33  # Mediapipe Pose has 33 landmarks
    keypoints = np.full((num_landmarks, 2), np.nan)
    confidences = np.full(num_landmarks, np.nan)

    if landmarks:
        for i, landmark in enumerate(landmarks[0]):
            pxl_x = landmark.x * width
            pxl_y = landmark.y * height
            keypoints[i, 0] = int(round(pxl_x))
            keypoints[i, 1] = int(round(pxl_y))
            confidences[i] = landmark.visibility

    return cam, keypoints, confidences


def inference_video(caps, projections, model_path, sweep_config=None):
    """
    Processes video frames from multiple cameras using multiprocessing, detects poses,
    scales landmarks to pixel coordinates, triangulates 3D points, and aggregates keypoint data.

    Parameters:
    - caps: List of tuples (camera_index, cv2.VideoCapture object)
    - projections: List of projection matrices corresponding to each camera
    - model_path: Path to the Mediapipe model file
    - sweep_config: Configuration for frame augmentation (optional)

    Returns:
    - keypoints_data: NumPy array of triangulated 3D keypoints
    - inference_time: NumPy array of inference times per frame
    - last_rgb_frame: Concatenated RGB frames from all cameras for the last processed frame
    """
    keypoints_data = []
    inference_time = []
    frame_number = 0
    num_cameras = len(caps)
    last_rgb_frame = None

    cam_indices = [cam for cam, _ in caps]
    cam_to_idx = {cam: idx for idx, cam in enumerate(cam_indices)}
    cap_status = {cam: True for cam in cam_indices}
    caps_dict = dict(caps)

    # Initialize the pool with the initializer
    pool = Pool(
        processes=num_cameras,
        initializer=init_worker,
        initargs=(model_path, sweep_config)
    )

    try:
        while True:
            frames = {}
            # Read one frame from each camera
            for cam in cam_indices:
                if cap_status[cam]:
                    ret, frame = caps_dict[cam].read()
                    if not ret:
                        cap_status[cam] = False  # End of video stream for this camera
                        frames[cam] = None
                        print(f"Camera {cam} has ended.")
                    else:
                        frames[cam] = frame
                else:
                    frames[cam] = None  # No more frames from this camera

            if not any(cap_status.values()):  # Check if all cameras are done
                print(f"All camera streams have ended at frame {frame_number}.")
                break

            start_time = time.time()

            # Prepare arguments for multiprocessing: list of (cam, frame)
            args_list = [(cam, frames[cam]) for cam in cam_indices]

            # Process frames in parallel
            results = pool.map(process_frame, args_list)

            # Collect results
            frame_keypoints = np.full((33, num_cameras, 2), np.nan)
            confidences = np.full((33, num_cameras), np.nan)

            for result in results:
                cam, keypoints, conf = result
                if cam is None:
                    continue
                idx = cam_to_idx.get(cam)
                if idx is None:
                    continue
                if keypoints is not None:
                    frame_keypoints[:, idx, :] = keypoints
                    confidences[:, idx] = conf

            # Triangulate 3D points
            points_3d = np.full((33, 3), np.nan)
            for landmark_idx in range(33):
                valid_indices = ~np.isnan(frame_keypoints[landmark_idx, :, 0])
                num_valid_cameras = np.sum(valid_indices)

                if num_valid_cameras >= 2:
                    # Collect 2D keypoints and projection matrices for valid cameras
                    keypoints_2d = frame_keypoints[landmark_idx, valid_indices, :]
                    valid_idxs = np.where(valid_indices)[0]
                    projections_valid = [projections[idx] for idx in valid_idxs]

                    try:
                        point_3d = DLT(projections_valid, keypoints_2d[np.newaxis, :, :])
                        points_3d[landmark_idx] = point_3d
                    except Exception as e:
                        print(f"Triangulation error for landmark {landmark_idx} in frame {frame_number}: {e}")
                else:
                    print(f"Not enough data to triangulate landmark {landmark_idx} in frame {frame_number}.")

            keypoints_data.append(points_3d)
            end_time = time.time()
            inference_time.append(end_time - start_time)
            frame_number += 1

    finally:
        pool.close()
        pool.join()
        # Release any remaining video captures
        for cam in cam_indices:
            if caps_dict[cam].isOpened():
                caps_dict[cam].release()
        cv2.destroyAllWindows()

    # Convert lists to numpy arrays
    keypoints_data = np.array(keypoints_data)
    inference_time = np.array(inference_time)

    return keypoints_data, inference_time, last_rgb_frame

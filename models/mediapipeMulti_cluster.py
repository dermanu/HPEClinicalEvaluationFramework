import cv2
cv2.setUseOptimized(True)
cv2.setNumThreads(2)
import numpy as np
import time
from multiprocessing import Pool, Manager
import mediapipe as mp
from mediapipe.tasks import python
from utils.frameAugmentation import FrameAugmentor
from mediapipe.tasks.python import vision
from models.dlt import DLT, weighted_DLT
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variables for each process
pose_landmarker = None
frame_augmentor = None


def init_worker(model_path, sweep_config):
    """
    Initializer for each worker process. Initializes PoseLandmarker and FrameAugmentor.
    """
    global pose_landmarker
    global frame_augmentor

    # Initialize PoseLandmarker
    pose_landmarker = vision.PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE
        )
    )

    # Initialize FrameAugmentor if needed
    if sweep_config and sweep_config.get('augmentation', 'none') != "none":
        frame_augmentor = FrameAugmentor()
    else:
        frame_augmentor = None


def process_frame(args):
    """
    Processes a single frame: augment, detect pose, and extract keypoints.
    """
    cam, frame = args
    if frame is None:
        return cam, None, None

    global pose_landmarker
    global frame_augmentor

    # Convert and rotate frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.rotate(rgb_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Apply frame augmentation if available
    if frame_augmentor is not None:
        rgb_frame = frame_augmentor.augment_frames(rgb_frame)

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
            pxl_x = int(round(pxl_x))
            pxl_y = int(round(pxl_y))
            keypoints[i, 0] = pxl_x
            keypoints[i, 1] = pxl_y
            confidences[i] = landmark.visibility

    return cam, keypoints, confidences


def inference_video(caps, projections, sweep_config=None):
    """
    Processes video frames from multiple cameras using multiprocessing, detects poses,
    scales landmarks to pixel coordinates, triangulates 3D points, and aggregates keypoint data.

    Parameters:
    - caps: List of tuples (camera_index, cv2.VideoCapture object)
    - projections: List of projection matrices corresponding to each camera
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

    # Prepare arguments for initializer
    model_path = 'models/pose_landmarker_full.task'

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
                        # End of video stream for this camera
                        cap_status[cam] = False
                        frames[cam] = None  # Set frame to None for inactive cameras
                        print(f"Camera {cam} has ended.")
                    else:
                        frames[cam] = frame
                else:
                    frames[cam] = None  # No more frames from this camera

            if not any(cap_status.values()):
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
            rgb_frames = {}

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
                    # Store the frame for visualization if needed
                    rgb_frames[cam] = frames[cam]

            # Triangulate 3D points
            points_3d = np.full((33, 3), np.nan)
            for landmark_idx in range(33):
                # Collect data for the current landmark across cameras
                valid_indices = ~np.isnan(frame_keypoints[landmark_idx, :, 0])
                num_valid_cameras = np.sum(valid_indices)

                if num_valid_cameras >= 2:
                    # Collect 2D keypoints and projection matrices for valid cameras
                    keypoints_2d = frame_keypoints[landmark_idx, valid_indices, :]
                    valid_idxs = np.where(valid_indices)[0]
                    projections_valid = [projections[idx] for idx in valid_idxs]

                    # Perform triangulation for the current landmark
                    try:
                        point_3d = DLT(projections_valid, keypoints_2d[np.newaxis, :, :])
                        points_3d[landmark_idx] = point_3d
                    except Exception as e:
                        print(f"Triangulation error for landmark {landmark_idx} in frame {frame_number}: {e}")
                else:
                    # Not enough data to triangulate this landmark
                    print(f"Not enough data to triangulate landmark {landmark_idx} in frame {frame_number}.")

            keypoints_data.append(points_3d)
            end_time = time.time()
            inference_time.append(end_time - start_time)
            frame_number += 1

            # Optional: Prepare the last RGB frame for display or saving
            rotated_frames = [np.rot90(frames[cam], k=-1) for cam in cam_indices if cam in frames]
            if rotated_frames:
                last_rgb_frame = np.concatenate(rotated_frames, axis=1)

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


import yaml

def load_projection_matrix(cam_index, file_path='P_values.yaml'):
    """
    Loads the projection matrix for a specified camera index from a YAML file.

    Parameters:
        cam_index (int): The index of the camera (0-based) to retrieve.
        file_path (str): Path to the YAML file containing the projection matrices.

    Returns:
        np.ndarray: The 3x4 projection matrix for the specified camera.
    """
    # Load the YAML file
    with open(file_path, 'r') as yaml_file:
        P_dict = yaml.safe_load(yaml_file)

    # Retrieve and return the projection matrix for the specified camera
    cam_key = f"Camera_{cam_index}"
    if cam_key in P_dict:
        return np.array(P_dict[cam_key])
    else:
        raise ValueError(f"Camera index {cam_index} is not in the file.")


def write_keypoints_to_disk(filename, kpts):
    fout = open(filename, 'w')

    for frame_kpts in kpts:
        for kpt in frame_kpts:
            if len(kpt) == 2:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ')
            else:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ' + str(kpt[2]) + ' ')

        fout.write('\n')
    fout.close()


if __name__ == '__main__':
    # Initialize video captures
    input_stream1 = '/home/emanu/Desktop/MoCap/segmented/par10/par10_Mov14_Cam0.avi'
    input_stream2 = '/home/emanu/Desktop/MoCap/segmented/par10/par10_Mov14_Cam4.avi'

    cap0 = cv2.VideoCapture(input_stream1)
    cap1 = cv2.VideoCapture(input_stream2)

    caps = [(0, cap0), (1, cap1)]  # Assign camera indices

    # Load projection matrices
    P0 = load_projection_matrix(0)
    P1 = load_projection_matrix(4)
    projections = [P0, P1]

    # Run inference
    keypoints_data, inference_time, last_rgb_frame = inference_video(caps, projections)

    # Save or process results as needed
    #this will create keypoints file in current working folder
    write_keypoints_to_disk('kpts_3D.dat', keypoints_data)

import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.frameAugmentation import FrameAugmentor
import mediapipe as mp
from mediapipe.tasks.python import vision
from models.dlt import DLT

# Load the pose landmarker model once to avoid reloading it multiple times
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='models/pose_landmarker_full.task'),
    running_mode=mp.tasks.vision.RunningMode.IMAGE)
PoseLandmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)


def process_frame(cap, frameaug=None, sweep_config=None):
    ret, frame = cap.read()
    # print(f"Frame read status: {ret}")  # Debug: Check if frame is read
    if not ret:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frameaug is not None and sweep_config is not None:
        rgb_frame = frameaug.augment_frames(rgb_frame, sweep_config)

    return rgb_frame


def detect_pose(rgb_frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = PoseLandmarker.detect(mp_image)
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
            # print(f"Processing frame {frame_number}")  # Debug: Loop iteration

            futures = [executor.submit(process_frame, cap[1], frameaug, sweep_config) for cap in caps]
            rgb_frames = []
            for future in as_completed(futures):
                try:
                    rgb_frame = future.result()
                    if rgb_frame is not None:
                        rgb_frames.append(rgb_frame)
                except Exception as e:
                    print(f"Error processing frame: {e}")

            frame_keypoints = np.zeros((33, num_cameras, 2))
            valid_detections = True  # Flag to check if all cameras detected poses

            if len(rgb_frames) == num_cameras:
                start_time = time.time()
                pose_futures = [executor.submit(detect_pose, rgb_frame) for rgb_frame in rgb_frames]
                for cam_idx, future in enumerate(as_completed(pose_futures)):
                    landmarks = future.result()
                    if landmarks:
                        for idx, pose_landmarks in enumerate(landmarks):
                            frame_keypoints[:, cam_idx, 0] = [landmark.x for landmark in pose_landmarks]
                            frame_keypoints[:, cam_idx, 1] = [landmark.y for landmark in pose_landmarks]
                    else:
                        valid_detections = False
                        print(f"No valid detection from camera {cam_idx}. Skipping triangulation.")
                        continue  # No need to continue if one camera fails to detect poses

            elif len(rgb_frames) == 0:
                break

            if valid_detections:
                # print(f"Triangulating for frame {frame_number}")
                points_3d = DLT(projections, frame_keypoints)
                # keypoints_data.append(points_3d.cpu().numpy())
                keypoints_data.append(points_3d)

                end_time = time.time()
                inference_time.append(end_time - start_time)
                frame_number += 1
                # print(f"Frame {frame_number} processed successfully.")

                if len(rgb_frames) >= 2:
                    last_rgb_frame = np.concatenate((rgb_frames), axis=1)
                elif len(rgb_frames) == 1:
                    last_rgb_frame = rgb_frames[0]
            else:
                keypoints_data.append(np.full((33, 3), np.nan))
                inference_time.append(np.nan)
                frame_number += 1
                print(f"Skipping frame {frame_number} due to invalid detections.")

    for _, cap in caps:
            cap.release()

    inference_time = np.array(inference_time)
    return np.array(keypoints_data), inference_time, last_rgb_frame
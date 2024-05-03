import cv2
import numpy as np
import time
import os
import sys

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
    except ImportError as e:
        raise e
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Configure OpenPose
params = {
    "model_folder": "path/to/openpose/models/",  # Replace with the path to the OpenPose models folder
    "hand": False,
    "face": False,
    "hand_detector": 2,
}

# Create OpenPose object
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


def process_video_openpose(input_video_path):
    # Open video file
    cap = cv2.VideoCapture(input_video_path)

    keypoints_data = []
    inference_time = []

    datum = op.Datum()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Record start time
        start_time = time.time()

        # Process the frame with OpenPose
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # Get keypoints from OpenPose output
        keypoints_frame = datum.poseKeypoints

        # Record end time
        end_time = time.time()

        inference_time.append(end_time - start_time)
        keypoints_data.append(keypoints_frame)

    cap.release()
    # Save keypoints data to NumPy array

    return np.array(keypoints_data), np.array(inference_time)

    # Release resources



def convert_openpose_to_mediapipe(openpose_keypoints):
    # Define mapping between OpenPose and MediaPipe keypoints
    mapping = {
        # Define your mapping here based on joint names or indices
    }

    mediapipe_keypoints = []

    for op_keypoint in openpose_keypoints:
        mediapipe_keypoints.append(op_keypoint)  # Placeholder, replace with actual conversion logic

    return mediapipe_keypoints

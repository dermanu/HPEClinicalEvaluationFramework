import cv2
import torch
import numpy as np
import time
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression


# Runs on GPU by default
def process_video_yolov7(input_video_path):
    # Load YOLOv7 model for pose estimation
    yolo_model = attempt_load("yolov7-pose.cfg", map_location=torch.device('cuda' if torch.cuda.is_available()
                                                                           else 'cpu'))
    stride = int(yolo_model.stride.max())  # Model stride

    # Open video file
    cap = cv2.VideoCapture(input_video_path)

    keypoints_data = []
    inference_time = []

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Record start time
        start_time = time.time()

        # Resize frame to model input size
        img = cv2.resize(frame, (640, 640))

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize and add batch dimension
        img = img / 255.0
        img = np.expand_dims(img, 0)

        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(img).float()

        # Run inference
        with torch.no_grad(): # Calculating gradients would cause a GPU memory leak
            pred = yolo_model(img_tensor)

        # Apply non-maximum suppression. Using same confidence thresholds as with MediaPipe
        pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)[0]

        # Record end time
        end_time = time.time()

        inference_time.append(end_time - start_time)

        # If pose detected, save keypoints to data list
        if pred is not None:
            keypoints = pred[:, :4].clone()
            keypoints[:, 2:] = keypoints[:, 2:] - keypoints[:, :2]  # Calculate width and height
            keypoints_data.append(adapt_format_yolov7_to_mediapipe(keypoints.cpu().numpy()))

        frame_number += 1

    # Save keypoints data to NumPy array
    return np.array(keypoints_data), inference_time

    # Release resources
    cap.release()


def adapt_format_yolov7_to_mediapipe(keypoints_yolov7):
    mapping = {
        # Define your mapping here based on joint names or indices
    }
    # Reshape YOLOv7 keypoints to [[x1, y1], [x2, y2], ..., [xn, yn]]
    return np.reshape(keypoints_yolov7, (-1, 2))

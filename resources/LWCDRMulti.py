import cv2
import numpy as np
import time
from utils.frameAugmentation import FrameAugmentor
from LWCDRmodel4 import GetModel, SII, FTL_inv, FTL
import tensorflow as tf


# Currently runs on CPU as per default.
def inference_video(caps, p_matrix, sweep_config=None,):
    # Load the pose landmarker model depending on size of input
    if len(caps) == 2:
        CDRnet = tf.keras.models.load_model('models/canonical_fusion_ep7500_2.h5',
                                            custom_objects={'SII': SII, '_dummy_loss': _dummy_loss, 'FTL_inv': FTL_inv,
                                                            'FTL': FTL})
    elif len(caps) == 3:
        CDRnet = tf.keras.models.load_model('models/canonical_fusion_ep7500_3.h5',
                                            custom_objects={'SII': SII, '_dummy_loss': _dummy_loss, 'FTL_inv': FTL_inv,
                                                            'FTL': FTL})
    elif len(caps) == 4:
        CDRnet = tf.keras.models.load_model('models/canonical_fusion_ep7500_4.h5',
                                            custom_objects={'SII': SII, '_dummy_loss': _dummy_loss, 'FTL_inv': FTL_inv,
                                                            'FTL': FTL})
    else:
        raise ValueError("Error: Not enough cameras defined")

    if sweep_config is not None:
        # Initialize frame augmentor
        frameaug = FrameAugmentor()

    keypoints_data = []
    inference_time = []
    frame_number = 0

    while caps.isOpened():
        ret, frame = caps.read()
        if not ret:
            break

        # Convert the BGR image to RGB - Is this really needed?
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if sweep_config is not None:
            # Augment frame
            rgb_frame = frameaug.augment_frames(rgb_frame, sweep_config)

        # Record start time
        start_time = time.time()

        # Prepare frame for the model
        input_frame = np.expand_dims(cv2.resize(rgb_frame, (256, 256)), axis=0)

        # Placeholder for projection matrices and inverse matrices
        pms = np.zeros((1, 4, 3, 4))  # Adjust based on your requirements
        pinvs = np.zeros((1, 4, 3, 4))  # Adjust based on your requirements

        # Model prediction
        x_v1_kpts, x_v2_kpts, x_v3_kpts, x_v4_kpts, recons = CDRnet.predict(
            [input_frame, input_frame, input_frame, input_frame,
             pms[0], pms[1], pms[2], pms[3],
             pinvs[0], pinvs[1], pinvs[2], pinvs[3]])

        kpts = np.stack([x_v1_kpts, x_v2_kpts, x_v3_kpts, x_v4_kpts], axis=1)

        # Record end time
        end_time = time.time()

        inference_time.append(end_time - start_time)

        # If keypoints detected, save to data list
        if kpts is not None:
            for view in range(4):
                frame_data = np.array([[i, kpt[0], kpt[1], kpt[2]] for i, kpt in enumerate(kpts[0, view])])
                keypoints_data.append(frame_data)

        frame_number += 1

    # Release resources
    caps.release()

    # Return keypoints data to NumPy array, save last frame for logging
    return np.array(keypoints_data), np.array(inference_time), rgb_frame
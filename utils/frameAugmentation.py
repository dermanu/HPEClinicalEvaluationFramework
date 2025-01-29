"""
Helper functions to augment frames durig evaluation.
"""

import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from pixellib.tune_bg import alter_bg

# Generate random ia seed
ia.seed(1)


def occlusion(frame):
    """
    Applies a random occlusion with Gaussian noise to the given image.

    Parameters:
    - frame (numpy.ndarray): The input image in the form of a NumPy array (height, width, channels).

    Returns:
    - frame (numpy.ndarray): The modified image with an occluded region filled with Gaussian noise.
    """

    h, w = frame.shape[:2]

    # Define the size of the occlusion
    occlusion_size = np.random.uniform(0.1, 0.25)

    # Compute the width and height of the occlusion
    cutout_width = int(w * occlusion_size)
    cutout_height = int(h * occlusion_size)

    # Randomly pick a position within the central region
    x1 = np.random.randint(w // 4, 3 * w // 4 - cutout_width)
    y1 = np.random.randint(h // 4, 3 * h // 4 - cutout_height)

    x2 = x1 + cutout_width
    y2 = y1 + cutout_height

    # Apply Gaussian noise to the cutout region
    noise = iaa.AdditiveGaussianNoise(scale=1 * 255).augment_image(frame)
    frame[y1:y2, x1:x2, :] = noise[y1:y2, x1:x2, :]

    return frame


def motion_blur(frame):
    """
    Applies a mild motion blur effect to the given image.

    Parameters:
    - frame (numpy.ndarray): The input image as a NumPy array with shape (height, width, channels).

    Returns:
    - numpy.ndarray: The image with a motion blur effect applied.
    """
    aug = iaa.imgcorruptlike.MotionBlur(severity=1)
    return aug(image=frame)


def overexposure(frame):
    """
    Simulates overexposure by increasing brightness.

    Parameters:
    - frame (numpy.ndarray): The input image as a NumPy array with shape (height, width, channels).

    Returns:
    - numpy.ndarray: The image with increased brightness to simulate overexposure.
   """
    # Add overexposure
    aug = iaa.color.MultiplyBrightness(0.2)
    return aug(image=frame)


def underexposure(frame):
    """
    Simulates underexposure by darkening the image.

    Parameters:
    - frame (numpy.ndarray): The input image as a NumPy array with shape (height, width, channels).

    Returns:
    - numpy.ndarray: The image with reduced brightness to simulate underexposure.
    """
    aug = iaa.BlendAlpha(
        (0.15),
        background=iaa.Multiply(0.2))
    return aug(image=frame)


def defocus(frame):
    """
    Simulates an out-of-focus effect using defocus blur.

    Parameters:
    - frame (numpy.ndarray): The input image as a NumPy array with shape (height, width, channels).

    Returns:
    - numpy.ndarray: The image with defocus blur applied.
    """
    aug = iaa.imgcorruptlike.DefocusBlur(severity=3)
    return aug(image=frame)


class FrameAugmentor:
    """
    A class for applying various augmentations to video frames before processing.

    Attributes:
    - change_bg (alter_bg): Background changer model for replacing backgrounds in frames.

    Methods:
    - augment_frames(frame, sweep_config): Applies augmentation to a frame based on the specified configuration.
    """
    def __init__(self):
        """
        Initializes the FrameAugmentor class.

        - Loads the background segmentation model using PixelLib.
        - Requires a pre-trained model file: 'utils/xception_pascalvoc.pb'.
        - The model file can be downloaded from:
          https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/xception_pascalvoc.pb
        """
        # Initiate BackgroundChanger
        self.change_bg = alter_bg(model_type="pb")
        # Load the model
        self.change_bg.load_pascalvoc_model('utils/xception_pascalvoc.pb')

    def augment_frames(self, frame, sweep_config):
        """
        Applies the specified augmentation to a single frame.

        Parameters:
        - frame (numpy.ndarray): The input image as a NumPy array with shape (height, width, channels).
        - sweep_config (dict): A dictionary containing augmentation settings.

        Returns:
        - numpy.ndarray: The augmented frame.
        """
        # Apply augmentation to each frame of input based on settings in sweep_config
        augmentation_type = sweep_config['augmentation']
        if augmentation_type == 'background':
            frame = self.change_bg.change_frame_bg(frame, 'utils/background.jpeg', detect="person")
        if augmentation_type == 'motion_blur':
            frame = motion_blur(frame)
        if augmentation_type == 'occlusion':
            frame = occlusion(frame)
        if augmentation_type == 'defocus':
            frame = defocus(frame)
        if augmentation_type == 'underexposure':
            frame = underexposure(frame)
        return frame


class CameraDesynchronizer:
    """
    A class to simulate camera desynchronization by introducing random frame offsets.

    Attributes:
    - rng (numpy.random.Generator): Random number generator for selecting frame offsets.

    Methods:
    - desynchronize(caps): Applies random frame offsets to a list of video capture objects.
    """
    def __init__(self):
        """
        Initializes the CameraDesynchronizer class.

        - Uses a fixed random seed (42) for reproducibility.
        - Generates random frame offsets to simulate desynchronization.
        """
        self.rng = np.random.default_rng(42)

    def desynchronize(self, caps):
        """
        Applies a small random frame offset to each camera feed.

        Parameters:
        - caps (list of tuples): A list of tuples (camera_id, cap), where:
             - camera_id (int): The identifier for the camera.
             - cap (cv2.VideoCapture): The OpenCV VideoCapture object.

        Returns:
        - list of tuples: A list of tuples (camera_id, cap) with the updated frame positions.
        """
        caps_offset = []

        for cam_id, cap in caps:  # Unpack the tuple (camera_id, cap) from caps
            frame_offset = self.rng.integers(low=0, high=1, size=1)  # This returns a NumPy array
            frame_offset = int(frame_offset[0])  # Convert the array to a scalar integer

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)  # Set the frame offset for the VideoCapture object
            caps_offset.append((cam_id, cap))  # Append the tuple (camera_id, cap) to the offset list

        return caps_offset
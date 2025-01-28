import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from pixellib.tune_bg import alter_bg

# Generate random ia seed
ia.seed(1)


def occlusion(frame):
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
    # Add motion blur
    aug = iaa.imgcorruptlike.MotionBlur(severity=1)
    return aug(image=frame)


def overexposure(frame):
    # Add overexposure
    aug = iaa.color.MultiplyBrightness(0.2)
    return aug(image=frame)


def underexposure(frame):
    # Add underexposure
    aug = iaa.BlendAlpha(
        (0.15),
        background=iaa.Multiply(0.2))
    return aug(image=frame)


def defocus(frame):
    # Add blur to simulate out of focus
    aug = iaa.imgcorruptlike.DefocusBlur(severity=3)
    return aug(image=frame)


class FrameAugmentor:
    def __init__(self):
        # Initiate BackgroundChanger
        self.change_bg = alter_bg(model_type="pb")
        # Load the model. Can be downloaded from
        # https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/xception_pascalvoc.pb
        self.change_bg.load_pascalvoc_model('utils/xception_pascalvoc.pb')

    def augment_frames(self, frame, sweep_config):
        """
        Augment video frames before being processed by model, according to sweep settings.
        :param sweep_config: Updated sweep config
        :param frames: [frame 0, frame 1, ...]
        :return: [augmented frame 0, augmented frame 1, ...]
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
    def __init__(self):
        self.rng = np.random.default_rng(42)

    # Desynchronize frames for multiple cameras by starting at different frames. This should be sufficient to simulate
    # different offset. However, not desynchronisation of long recordings due to clock drift.
    def desynchronize(self, caps):
        # Open video file at different offset frames
        caps_offset = []

        for cam_id, cap in caps:  # Unpack the tuple (camera_id, cap) from caps
            frame_offset = self.rng.integers(low=0, high=1, size=1)  # This returns a NumPy array
            frame_offset = int(frame_offset[0])  # Convert the array to a scalar integer

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)  # Set the frame offset for the VideoCapture object
            caps_offset.append((cam_id, cap))  # Append the tuple (camera_id, cap) to the offset list

        return caps_offset
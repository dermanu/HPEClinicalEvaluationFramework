import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from pixellib.tune_bg import alter_bg

# Generate random ia seed
ia.seed(1)


def occlusion(frame):
    # Create moving occlusion of 10-20% of the image size and fill it with gaussian noise.
    aug = iaa.Cutout(size=(0.1, 0.2), fill_mode="gaussian", fill_per_channel=True)

    return aug(image=frame)


def motion_blur(frame):
    # Add motion blur
    aug = iaa.imgcorruptlike.MotionBlur(severity=1)

    return aug(image=frame)


def overexposure(frame):
    # Add overexposure
    aug = iaa.imgcorruptlike.Brightness(severity=2)

    return aug(image=frame)


def underexposure(frame):
    # Add underexposure
    aug = iaa.GammaContrast(0.5)

    return aug(image=frame)


def defocus(frame):
    # Add blur to simulate out of focus
    aug = iaa.imgcorruptlike.DefocusBlur(severity=1)

    return aug(image=frame)


class FrameAugmentor:
    def __init__(self):
        # Initiate BackgroundChanger
        self.change_bg = alter_bg(model_type="pb")
        self.change_bg.load_pascalvoc_model("xception_pascalvoc.pb")

    def change_background(self, frame, background):
        if background == 'home':
            aug_frame = self.change_bg.change_frame_bg(frame, 'home.jpg', detect="person")
        elif background == 'hospital':
            aug_frame = self.change_bg.change_frame_bg(frame, 'hospital.jpg', detect="person")
        elif background == 'outdoor':
            aug_frame = self.change_bg.change_frame_bg(frame, 'outdoor.jpg', detect="person")
        elif background == 'outdoor':
            aug_frame = self.change_bg.change_frame_bg(frame, 'people.jpg', detect="person")
        elif background == 'none':
            aug_frame = frame
        else:
            raise ValueError("Choose a valid background sweep parameter")

        return aug_frame

    def augment_frames(self, frames, sweep_config):
        """
        Augment video frames before being processed by model, according to sweep settings.
        :param sweep_config: Updated sweep config
        :param frames: [frame 0, frame 1, ...]
        :return: [augmented frame 0, augmented frame 1, ...]
        """

        # Apply augmentation to each frame of input based on settings in sweep_config
        frames_aug = []
        for frame in frames:
            if sweep_config['background'] != 'none':
                frame = self.change_background(frame, sweep_config['background'])
            if sweep_config['motion_blur']:
                frame = motion_blur(frame)
            if sweep_config['occlusion']:
                frame = occlusion(frame)
            if sweep_config['defocus']:
                frame = defocus(frame)
            if sweep_config['underexposure']:
                frame = underexposure(frame)
            if sweep_config['overexposure']:
                frame = underexposure(frame)

            frames_aug.append(frame)

        return frames_aug


class CameraDesynchronizer:
    def __init__(self):
        self.rng = np.random.default_rng(1)

    # Desynchronize frames for multiple cameras by starting at different frames. This should be sufficient to simulate
    # different offset. However, not desynchronisation of long recordings due to clock drift.
    def desynchronize(self, caps):
        # Open video file at different offset frames
        caps_offset = []
        for cap in caps:
            frame_offset = self.rng.integers(low=0, high=2, size=1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)
            caps_offset.append(cap)

        return caps_offset

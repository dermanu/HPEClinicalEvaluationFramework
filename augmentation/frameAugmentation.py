import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from pixellib.tune_bg import alter_bg


# Generate random seed for imgaug
ia.seed(1)


# Add blur to simulate out of focus
def defocus(frame):
    aug = iaa.imgcorruptlike.DefocusBlur(severity=1)

    return aug(image=frame)


# Add underexposure
def underexposure(frame):
    aug = iaa.GammaContrast(0.5)

    return aug(image=frame)


# Add overexposure
def overexposure(self, frame):
    aug = iaa.imgcorruptlike.Brightness(severity=2)

    return aug(image=frame)


# Add motion blur
def motion_blur(frame):
    aug = iaa.imgcorruptlike.MotionBlur(severity=1)

    return aug(image=frame)


# Create moving occlusion of 10-20% of the image size and fill it with gaussian noise.
def occlusion(frame):
    aug = iaa.Cutout(size=(0.1, 0.2), fill_mode="gaussian", fill_per_channel=True)

    return aug(image=frame)


class BackgroundChanger:
    def __init__(self):
        self.change_bg = alter_bg(model_type="pb")
        self.change_bg.load_pascalvoc_model("xception_pascalvoc.pb")

# Changes the background of a frame according to the chosen background type
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


class CameraDesynchronizer:
    def __init__(self):
        self.rg = np.random.default_rng(1)

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


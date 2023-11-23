"""
This script applySkeletonMorphing allows to morph keypoints form one model to another during training.
Example Usage:
    model_path = 'path/to/your/model.pt'
    morphing_model = ApplySkeletonMorphing(model_path)
    input_pose = torch.tensor(...)  # Your input pose tensor
    morphed_pose = morphing_model.apply_morphing(input_pose)
"""

import torch

class ApplySkeletonMorphing:
    def __init__(self, model_path):
        """
        Initialize the Skeleton Morphing Model by loading the model from the specified path.

        Parameters:
        - model_path: Path to the saved skeleton morphing model.
        """
        self.model = torch.load(model_path)
        self.model.eval()

    def apply_morphing(self, input_pose):
        """
        Apply skeleton morphing to the input pose using the loaded model.

        Parameters:
        - input_pose: Input pose to be morphed.

        Returns:
        - morphed_pose: Pose after applying skeleton morphing.
        """
        with torch.no_grad():
            morphed_pose = self.model(input_pose)
        return morphed_pose
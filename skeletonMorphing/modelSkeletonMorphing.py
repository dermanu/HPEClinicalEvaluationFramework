"""
The function modelSkeletonMorphing is a trainable model to morph keypoints of one dataset to another for comparison of
different algorithms.
It is based on the work of Bastian Wandt (https://github.com/bastianwandt/CanonPose/tree/main).
"""

import torch.nn as nn


class Synthesizer(nn.Module):
    def __init__(self):
        super(Synthesizer, self).__init__()

        # Upscaling layer to transform input of size 48 to 2304
        self.upscale = nn.Linear(48, 2304)

        # Residual blocks for common features
        self.res_common = ResBlock()

        # Residual blocks for pose processing
        self.res_pose1 = ResBlock()
        self.res_pose2 = ResBlock()
        self.res_pose3 = ResBlock()

        # Linear layer for morphing pose information back to 48 dimensions
        self.pose_morph = nn.Linear(2304, 48)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.20)  # Add dropout layer with probability 0.25

    def forward(self, x):
        # Upscaling the input
        xu = self.upscale(x)

        # Pose processing path
        xp = self.dropout(nn.LeakyReLU()(self.res_pose1(xu)))
        xp = self.dropout(nn.LeakyReLU()(self.res_pose2(xp)))
        xp = self.dropout(nn.LeakyReLU()(self.res_pose3(xp)))

        # Adding morphed pose information back to the input
        x_pose = x + self.pose_morph(xp)

        return x_pose


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()

        # Two linear layers for the residual block
        self.l1 = nn.Linear(2304, 2304)
        self.l2 = nn.Linear(2304, 2304)
        self.l3 = nn.Linear(2304, 2304)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.20)  # Add dropout layer with probability 0.25

    def forward(self, x):
        inp = x

        # Leaky ReLU activation for the first linear layer
        x = self.dropout(nn.LeakyReLU()(self.l1(x)))

        # Leaky ReLU activation for the second linear layer
        x = self.dropout(nn.LeakyReLU()(self.l2(x)))

        # Leaky ReLU activation for the third linear layer
        x = self.dropout(nn.LeakyReLU()(self.l3(x)))

        # Adding the residual connection
        x += inp

        return x

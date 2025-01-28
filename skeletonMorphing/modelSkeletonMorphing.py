"""
The function modelSkeletonMorphing is a trainable model to morph keypoints of one dataset to another for comparison of
different algorithms.
It is inspired by the work of Bastian Wandt (https://github.com/bastianwandt/CanonPose/tree/main).
"""

import torch.nn as nn

class Synthesizer(nn.Module):
    def __init__(self, dropout_rate=0.0, layer_size=1024):
        super(Synthesizer, self).__init__()

        # Upscaling layer to transform input of size 48 to 1024
        self.upscale = nn.Linear(3*16, layer_size)

        # Create a list of residual blocks based on num_blocks
        self.res_block1 = ResBlock(dropout_rate, layer_size)
        self.res_block2 = ResBlock(dropout_rate, layer_size)

        # Linear layer for morphing pose information back to 48 dimensions
        self.pose_morph = nn.Linear(layer_size, 3*16)

        # Save dropout rate for use in forward method if needed
        self.dropout = nn.Dropout(p=dropout_rate)

        # Make variables available
        self.dropout_rate = dropout_rate
        self.layer_size = layer_size


    def forward(self, x):
        # Upscaling the input
        x_up = self.upscale(x)

        # Pass through the residual blocks
        x_res = nn.LeakyReLU()(self.res_block1(x_up))
        if self.dropout_rate > 0:
            x_res = self.dropout(x_res)
        x_res = nn.LeakyReLU()(self.res_block2(x_res))
        if self.dropout_rate > 0:
            x_res = self.dropout(x_res)

        # Morphing back to original dimensions and adding skip connection
        x_pose = x + self.pose_morph(x_res)
        return x_pose


class ResBlock(nn.Module):
    def __init__(self, dropout_rate=0.0, layer_size=1024):
        super(ResBlock, self).__init__()
        # Two linear layers for the residual block
        self.l1 = nn.Linear(layer_size, layer_size)
        self.l2 = nn.Linear(layer_size, layer_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        input = x
        out = nn.LeakyReLU()(self.l1(x))
        if self.dropout.p > 0:
            out = self.dropout(out)

        out = nn.LeakyReLU()(self.l2(out))
        if self.dropout.p > 0:
            out = self.dropout(out)
        out += input
        return out

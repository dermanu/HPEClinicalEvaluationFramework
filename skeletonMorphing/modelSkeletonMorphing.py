"""
The function modelSkeletonMorphing is a trainable model to morph keypoints of one dataset to another for comparison of
different algorithms.
It is based on the work of Bastian Wandt (https://github.com/bastianwandt/CanonPose/tree/main).
"""
import torch
import torch.nn as nn


class Synthesizer(nn.Module):
    def __init__(self, dropout_rate):
        super(Synthesizer, self).__init__()

        # Upscaling layer to transform input of size 48 to 2304
        self.upscale = nn.Linear(3*16*6, 1024)


        # Residual blocks for pose processing
        self.res_pose1 = ResBlock()
        #self.bn1 = nn.BatchNorm1d(2048)
        self.res_pose2 = ResBlock()
        #self.bn2 = nn.BatchNorm1d(2048)


        # Linear layer for morphing pose information back to 48 dimensions
        self.pose_morph = nn.Linear(1024, 3*16)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout_rate)  # Add dropout layer with probability 0.20

    def forward(self, x):
        # Upscaling the input
        #print(x.shape)
        xu = nn.LeakyReLU()(self.upscale(x))

        # Pose processing path
        xp = self.dropout(nn.LeakyReLU()(self.res_pose1(xu)))
        xp = self.dropout(nn.LeakyReLU()(self.res_pose2(xp)))

        x = x.view(-1, 6, 48)

        return self.pose_morph(xp)


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()

        # Two linear layers for the residual block
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        #self.bn1 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 1024)
        #self.bn2 = nn.BatchNorm1d(1024)

        #self.l1 = nn.Linear(2048, 2048)
        #self.l2 = nn.Linear(2048, 2048)
        #self.l3 = nn.Linear(2048, 2048)
        # Dropout layer for regularization
        #self.dropout = nn.Dropout(p=0.00)  # Add dropout layer with probability 0.20

    def forward(self, x):
        inp = x

        # Leaky ReLU activation for the first linear layer
        #x = self.dropout(nn.LeakyReLU()(self.l1(x)))

        # Leaky ReLU activation for the second linear layer
        #x = self.dropout(nn.LeakyReLU()(self.l2(x)))

        # Leaky ReLU activation for the third linear layer
        #x = self.dropout(nn.LeakyReLU()(self.l3(x)))

        # Adding the residual connection
        #x += inp
        residual = x
        out = nn.LeakyReLU()(self.fc1(x))
        out = nn.LeakyReLU()(self.fc2(out))
        out = nn.LeakyReLU()(self.fc3(out))
        out += residual
        out = nn.LeakyReLU()(out)

        return out

"""
The function modelSkeletonMorphing is a trainable model to morph keypoints of one dataset to another for comparison of
different algorithms.
It is based on the work of Bastian Wandt (https://github.com/bastianwandt/CanonPose/tree/main).
"""
import torch
import torch.nn as nn


class Synthesizer(nn.Module):
    def __init__(self):
        super(Synthesizer, self).__init__()

        # Upscaling layer to transform input of size 48 to 2304
        self.upscale = nn.Linear(3*16*6, 2048)

        # Residual blocks for common features
        self.res_common = ResBlock()

        # Residual blocks for pose processing
        self.res_pose1 = ResBlock()
        self.bn1 = nn.BatchNorm1d(2048)
        self.res_pose2 = ResBlock()
        self.bn2 = nn.BatchNorm1d(2048)
        self.res_pose3 = ResBlock()
        self.bn3 = nn.BatchNorm1d(2048)
        self.res_pose4 = ResBlock()
        #self.res_pose5 = ResBlock()
        #self.res_pose6 = ResBlock()

        # Linear layer for morphing pose information back to 48 dimensions
        self.pose_morph = nn.Linear(2048, 3*16)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.2)  # Add dropout layer with probability 0.20

    def forward(self, x, dist):
        # Upscaling the input
        #print(x.shape)
        xu = self.upscale(x)

        # Pose processing path
        xp = self.dropout(nn.LeakyReLU()(self.res_pose1(xu)))
        #xp = self.bn1(xp)
        #xp = self.dropout(nn.LeakyReLU()(self.res_pose2(xp)))
        #xp = self.bn2(xp)
        #xp = self.dropout(nn.LeakyReLU()(self.res_pose3(xp)))
        #xp = self.bn3(xp)
        #xp = self.dropout(nn.LeakyReLU()(self.res_pose4(xp)))
        #xp = self.dropout(nn.LeakyReLU()(self.res_pose5(xp)))
        #xp = self.dropout(nn.LeakyReLU()(self.res_pose6(xp)))

        # Adding morphed pose information back to the input
        x = x.view(-1, 6, 48)
        dist = dist.view(-1, 6, 16)

        normalized_confidences = torch.nn.functional.softmax(dist, dim=1)
        #print("Normalized confidences", torch.nn.functional.softmax(dist, dim=0))
        #print("Normalized confidences",torch.nn.functional.softmax(dist, dim=1))
        #print("Normalized confidences", torch.nn.functional.softmax(dist, dim=2))
        # Calculate the weighted mean for each joint
        weighted_means = torch.zeros(x.size(0),16, 3).cuda()  # Initialize tensor to store the results
        #print(torch.mean(dist, dim=2))

        if torch.any(torch.mean(dist, dim=1) < 0.7):
            print(dist)
            print(torch.mean(dist, dim=1))
            raise Exception("Not sure enough")
        if False:
            for camera in range(6):
                # Extract the coordinates and normalized confidences for the current joint from all cameras
                joint_coordinates = x.view(-1, 6, 16, 3)[:, camera, :]
                joint_confidences = normalized_confidences[:, camera]
                #print(joint_coordinates.shape, joint_confidences.shape)

                # Calculate the weighted mean for the current joint
                for i in range(3):
                    weighted_means[:, :, i] += joint_coordinates.view(-1, 16, 3)[:, :, i] * joint_confidences[:, :]
                #weighted_mean = torch.sum(joint_coordinates.reshape(-1, 16, 3) * joint_confidences, dim=1)
                #print(joint_coordinates * joint_confidences)
                #print(weighted_means)
                # Store the weighted mean in the result tensor


        #print(weighted_means)
        #x_pose = weighted_means.reshape(-1, 48) + self.pose_morph(xp)

        return self.pose_morph(xp)
        return x_pose


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()

        # Two linear layers for the residual block
        self.l1 = nn.Linear(2048, 2048)
        self.l2 = nn.Linear(2048, 2048)
        self.l3 = nn.Linear(2048, 2048)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.00)  # Add dropout layer with probability 0.20

    def forward(self, x):
        inp = x

        # Leaky ReLU activation for the first linear layer
        x = self.dropout(nn.LeakyReLU()(self.l1(x)))

        # Leaky ReLU activation for the second linear layer
        x = self.dropout(nn.LeakyReLU()(self.l2(x)))

        # Leaky ReLU activation for the third linear layer
        #x = self.dropout(nn.LeakyReLU()(self.l3(x)))

        # Adding the residual connection
        x += inp

        return x

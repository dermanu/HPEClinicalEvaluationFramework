"""
This script trainSkeletonMorphing trains the model modelSkeletonMorphing based on the ground throuth of two different
dataset.
The morphing network should be trained one subject of each dataset. Subject one then has to be excluded for all other
experiments to not include ground truth data.
It is based on the work of Bastian Wandt (https://github.com/bastianwandt/CanonPose/tree/main and
https://arxiv.org/pdf/2011.14679.pdf).
"""

import torch.optim
from torch.utils import data
import torch.optim as optim
import modelSkeletonMorphing
from losses import print_losses
from types import SimpleNamespace
import torch.nn as nn
from utils.readDataset import ReadDatasetFiles


# Configuration settings using SimpleNamespace
config = SimpleNamespace()
config.learning_rate = 0.0001
config.BATCH_SIZE = 32
config.N_epochs = 100

# Folder containing data
data_folder = '/home/emanu/Desktop/SegmentedData'

# Parameters for training
par = [4]
mov = list(range(1, 18))
cam = list(range(0, 6))
model_type = 'openpose'

# Data file for training
config.datafile = data_folder + 'h36m_train_mpi_skeleton_pred.pickle'

# Creating dataset and data loader
my_dataset = ReadDatasetFiles(data_folder, par, mov, cam, model_type)

train_loader = data.DataLoader(my_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

# Initializing the model (Synthesizer) and moving it to GPU
model = modelSkeletonMorphing.Synthesizer().cuda()

# Mean Squared Error Loss
mse_loss = nn.MSELoss()

# Number of epochs
N_epochs = 100

# Parameters for optimization
params = list(model.parameters())  # + list(dec.parameters())

# Setting anomaly detection for autograd
optimizer = optim.Adam(params, lr=config.learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 95], gamma=0.1)

# Setting anomaly detection for autograd
torch.autograd.set_detect_anomaly(True)

# Namespace to store losses during training
losses = SimpleNamespace()
losses_mean = SimpleNamespace()

# List of camera IDs
cams = ['54138969', '55011271', '58860488', '60457274']
all_cams = ['cam0', 'cam1', 'cam2', 'cam3']

# Training loop
for epoch in range(N_epochs):

    for i, sample in enumerate(train_loader):

        # Extracting 2D poses and ground truth 2D poses
        poses_2d = {key: sample[key] for key in all_cams}
        poses_2dgt = sample['p2d_gt']

        # Creating tensors for input and output poses
        inp_poses = torch.zeros((poses_2d['cam0'].shape[0] * 4, 32)).cuda()
        output_poses = torch.zeros((poses_2dgt['cam0'].shape[0] * 4, 32)).cuda()

        cnt = 0

        # Flattening input poses and confidences for all cameras
        for b in range(poses_2d['cam0'].shape[0]):
            for c_idx, cam in enumerate(poses_2d):
                inp_poses[cnt] = poses_2d[cam][b]
                output_poses[cnt] = poses_2dgt[cam][b]
                cnt += 1

        # Forward pass through the model
        pred_poses = model(inp_poses)

        # Calculating MSE loss
        losses.loss = mse_loss(pred_poses, output_poses)

        # Backward pass and optimization step
        optimizer.zero_grad()
        losses.loss.backward()
        optimizer.step()

        # Storing losses for printing
        for key, value in losses.__dict__.items():
            if key not in losses_mean.__dict__.keys():
                losses_mean.__dict__[key] = []

            losses_mean.__dict__[key].append(value.item())

        # Printing losses every 100 iterations
        if not i % 100:
            print_losses(epoch, i, len(my_dataset) / config.BATCH_SIZE, losses_mean.__dict__, print_keys=not (i % 1000))

            # Resetting losses_mean for the next set of iterations
            losses_mean = SimpleNamespace()

    # Saving the model after each epoch
    torch.save(model, 'models/model_skeleton_morph_S1_gh.pt')

# Training complete
print('done')

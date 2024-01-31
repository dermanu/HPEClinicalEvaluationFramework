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
from utils.readDataset2 import ReadDatasetFiles
import wandb
import numpy as np
from utils.plot_keypoints import plot_3d_keypoints, plot_3d_keypoints_all


# Configuration settings using SimpleNamespace
config = SimpleNamespace()
config.learning_rate = 0.0001
config.BATCH_SIZE = 32
config.N_epochs = 100
config.log_interval = 100
config.weight_decay = 1e-5
last_loss_mean = 100000

# Sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.0001, 0.00001]
        },
        'BATCH_SIZE': {
            'values': [8, 16, 32]
        },
        'weight_decay': {
            'values': [1e-4, 1e-5, 1e-6]
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 10
    }
}

# WandB – Initialize a new run
wandb.init(project="skeleton-morphing", config=config)


# Folder containing data
data_folder = '/home/emanu/Desktop/SegmentedData'

# Parameters for training
par = [4]
mov = list(range(1, 18))
cam = list(range(0, 6))
model_type = 'mediapipe'
num_cam = 6

# Data file for training
# config.datafile = data_folder + 'h36m_train_mpi_skeleton_pred.pickle'

# Creating dataset and data loader
# my_dataset = ReadDatasetFiles(data_folder, par, mov, cam, model_type)
# torch.save(my_dataset, 'par4_mediapipe_test2.pth')
my_dataset = torch.load('par4_mediapipe_test.pth')
print('Data loader created')
train_loader = data.DataLoader(my_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

# Initializing the model (Synthesizer) and moving it to GPU
model = modelSkeletonMorphing.Synthesizer().cuda()

wandb.watch(model, log_freq=100)

# Mean Squared Error Loss
mse_loss = nn.MSELoss()

# Number of epochs
N_epochs = 100

# Parameters for optimization
params = list(model.parameters())  # + list(dec.parameters())

# Setting anomaly detection for autograd
optimizer = optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 95], gamma=0.1)

# Setting anomaly detection for autograd
torch.autograd.set_detect_anomaly(True)

# Namespace to store losses during training
losses_mean = []
batch_idx = 0

# Training loop
for epoch in range(N_epochs):
    for batch in train_loader:
        batch_idx += 1
        # Access data for each batch
        pose_gt_batch = batch['pose_gt']
        pose_inf_batch = batch['pose_inf']

        # Creating tensors for input and output poses
        inp_poses = pose_inf_batch.view(-1, pose_inf_batch.size(1) * pose_inf_batch.size(2)).cuda().float()  # batches/frames x cams, keypoints x 3
        output_poses = pose_gt_batch.view(-1, pose_gt_batch.size(1) * pose_gt_batch.size(2)).cuda().float()

        # Forward pass through the model
        pred_poses = model(inp_poses)

        # Calculating MSE loss
        loss = mse_loss(pred_poses, output_poses)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_mean.append(loss.item())


    wandb.log({"loss": np.mean(losses_mean)})
    prediction = pred_poses.view(-1, pose_gt_batch.size(1), pose_gt_batch.size(2)).cpu().detach().numpy()[0]
    ground_truth = pose_gt_batch.cpu().detach().numpy()[0]
    hpe_truth = pose_inf_batch.cpu().detach().numpy()[0]
    plot_3d_keypoints(prediction, 'mediapipe', 'morphed')
    plot_3d_keypoints(hpe_truth, 'mediapipe', 'ground_truth')
    plot_3d_keypoints(hpe_truth, 'mediapipe', 'hpe_truth')
    plot_3d_keypoints_all(prediction, ground_truth, hpe_truth, 'mediapipe')
    wandb.log({"epoch": epoch})

    # Saving the model after each epoch
    print('Finished epoch ' + str(epoch) + ' of ' + str(N_epochs) + ' with loss ' + str(np.mean(losses_mean)))
    if np.mean(losses_mean) < last_loss_mean:
        last_loss_mean = np.mean(losses_mean)
        torch.save(model, 'models/model_skeleton_morph_par4_mediapipe.pt')

    losses_mean = []

# Training complete
print('done')


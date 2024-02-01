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
import time
import torch.nn as nn
from utils.readDataset2 import ReadDatasetFiles
import wandb
import numpy as np
from utils.plot_keypoints import plot_3d_keypoints, plot_3d_keypoints_all
import tqdm


# Sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'epoch_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.0001, 0.00001]
        },
        'BATCH_SIZE': {
            'values': [16, 32, 64]
        },
        'weight_decay': {
            'values': [1e-4, 1e-5, 1e-6]
        },
        'epochs': {
            'value': 100
        },
        'datafolder': {
            'value': '/home/emanu/Desktop/SegmentedData'
        },
        'par': {
            'value': [4]
        },
        'mov': {
            'value': list(range(1, 18))
        },
        'cam': {
            'value': list(range(0, 6))
        },
        'model_type': {
            'value': 'mediapipe'
        },
        'num_cam': {
            'value': 6
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3,
        'eta': 3
    }
}

# Initialize sweep by passing in config.
# (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_config, project="SkeletonMorphingSweep")


def data_loader(data_config):
    # Creating dataset and data loader
    # my_dataset = ReadDatasetFiles(data_folder, config.par, config.mov, config.cam, config.model_type)
    # torch.save(my_dataset, 'par4_mediapipe_test2.pth')
    my_dataset = torch.load('morph_dataset/par4_mediapipe_test.pth')
    train_loader = data.DataLoader(my_dataset, batch_size=data_config.BATCH_SIZE, shuffle=True, num_workers=8,
                                   pin_memory=True)

    return train_loader


def train(model, train_loader, optimizer):
    # Iterate through batches
    losses = 0
    for step, batch in enumerate(tqdm.tqdm(train_loader, desc="Training progress", leave=False)):
        # Access data for each batch
        pose_gt_batch = batch['pose_gt']
        pose_inf_batch = batch['pose_inf']

        # Creating tensors for input and output poses
        inp_poses = pose_inf_batch.view(-1, pose_inf_batch.size(1) * pose_inf_batch.size(
            2)).cuda().float()  # batches/frames x cams, keypoints x 3
        output_poses = pose_gt_batch.view(-1, pose_gt_batch.size(1) * pose_gt_batch.size(2)).cuda().float()

        # Forward pass through the model
        pred_poses = model(inp_poses)

        # Calculating MSE loss
        loss = nn.functional.mse_loss(pred_poses, output_poses)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Append loss
        losses += (loss.item())

        # Log the loss of each batch
        wandb.log({"batch_loss": loss.item(), "batch": step+1})
        print

    return losses / len(train_loader), pred_poses, pose_gt_batch, pose_inf_batch


def main(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # Load model
        model = modelSkeletonMorphing.Synthesizer().cuda()
        # wandb.watch(model, log_freq=1000)

        # Parameters for optimization
        params = list(model.parameters())  # + list(dec.parameters())

        # Setting anomaly detection for autograd
        optimizer = optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)

        # Setting anomaly detection for autograd
        torch.autograd.set_detect_anomaly(True)

        # Load dataset
        dataset = data_loader(config)

        # Start loss
        last_loss = 1000000

        # Training loop
        for epoch in range(config.epochs):
            time.sleep(15)
            # Training the model
            losses, pred_poses, pose_gt_batch, pose_inf_batch = train(model, dataset, optimizer)

            # Logging the loss and epoch
            wandb.log({"epoch_loss": np.sqrt(np.mean(losses)), "epoch": epoch+1})

            # Plotting the keypoints and logging them
            prediction = pred_poses.view(-1, pose_gt_batch.size(1), pose_gt_batch.size(2)).cpu().detach().numpy()[0]
            ground_truth = pose_gt_batch.cpu().detach().numpy()[0]
            hpe_truth = pose_inf_batch.cpu().detach().numpy()[0]
            plot_3d_keypoints(prediction, 'mediapipe', 'morphed', epoch)
            plot_3d_keypoints(hpe_truth, 'mediapipe', 'ground_truth', epoch)
            plot_3d_keypoints(hpe_truth, 'mediapipe', 'hpe_truth', epoch)
            plot_3d_keypoints_all(prediction, ground_truth, hpe_truth, 'mediapipe', epoch)

            # Print loss and epoch
            print('Finished epoch ' + str(epoch+1) + ' of ' + str(config.epochs) + ' with loss ' + str(np.mean(losses)))

            # Save so far best model
            if np.mean(losses) < last_loss:
                model_path = 'models/morph_' + config.model_type + '_' + sweep_id + '.pth'
                torch.save(model.state_dict(), model_path)
                last_loss = np.mean(losses)

            # Save best model to wandb
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)


# Start sweep job.
wandb.agent(sweep_id, function=main, count=5)
# Training complete
print('done')
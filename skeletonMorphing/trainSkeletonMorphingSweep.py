"""
This script trainSkeletonMorphing trains the model modelSkeletonMorphing based on the ground thruth of two different
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
import torch.nn as nn
import wandb
import numpy as np
from utils.plotKeypoints import plot_3d_keypoints, plot_3d_keypoints_all
from skeletonMorphing.trainSkeletonMorphing import NetworkTrainer, MPJPELoss, load_train_test_all
import tqdm


# Sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'validation_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-2, 1e-3, 1e-4]
        },
        'BATCH_SIZE': {
            'values': [16, 32, 64]
        },
        'weight_decay': {
            'values': [1e-1, 1e-2, 1e-3]
        },
        'epochs': {
            'values': [60, 80, 100]
        },
        'dropout': {
            'values': [0, 0.1, 0.2]
        },
        'num_blocks': {
            'values': [2, 3]
        },
        'layer_size': {
            'values': [1024, 2048]
        },
        'datafolder': {
            'value': "E:\MoCap"
        },
        'pars': {
            'values': [11, 25]
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
        'min_iter': 7,
        'eta': 3
    }
}

# Initialize sweep by passing in config.
sweep_id = wandb.sweep(sweep=sweep_config, project="SkeletonMorphingSweep")

# RMSE loss function
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

# Mean Per Joint Position Error Loss
class MPJPELoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(skeletons1, skeletons2):
        coordinate_abs_diff = torch.abs(skeletons1 - skeletons2)
        mae_per_coordinate = torch.mean(coordinate_abs_diff, dim=0)
        mae_per_joint_position = torch.mean(mae_per_coordinate, dim=1)
        overall_mae = torch.mean(mae_per_joint_position)
        return overall_mae

# Trainings data loader
def train_data_loader(data_config):
    """
    Creating dataset and data loader for training
    :param data_config: Configuration for the data loader
    """
    # my_dataset = ReadDatasetFiles(data_folder, config.par, config.mov, config.cam, config.model_type)
    # torch.save(my_dataset, 'par4_mediapipe_test2.pth')
    my_dataset1 = torch.load('morph_dataset/par4_mediapipe_test.pth')
    my_dataset2 = torch.load('morph_dataset/par5_mediapipe_test.pth')
    #my_dataset = torch.utils.data.ConcatDataset([my_dataset1, my_dataset2])
    train_loader = data.DataLoader(my_dataset1, batch_size=data_config.BATCH_SIZE, shuffle=True, num_workers=8,
                                   pin_memory=True)
    return train_loader

# Validation data loader
def validation_data_loader(data_config):
    """
    Creating dataset and data loader
    :param data_config: Configuration for the data loader
    """
    # Save dataset in pytorch model
    # my_dataset = ReadDatasetFiles(data_folder, config.par, config.mov, config.cam, config.model_type)
    # torch.save(my_dataset, 'par4_mediapipe_test2.pth')
    my_dataset = torch.load('morph_dataset/par5_mediapipe_test.pth')
    validation_loader = data.DataLoader(my_dataset, batch_size=data_config.BATCH_SIZE, shuffle=False, num_workers=8,
                                  pin_memory=True)
    return validation_loader

# Test data loader
def test_data_loader(data_config):
    """
    Creating dataset and data loader
    :param data_config: Configuration for the data loader
    """
    # Save dataset in pytorch model
    # my_dataset = ReadDatasetFiles(data_folder, config.par, config.mov, config.cam, config.model_type)
    # torch.save(my_dataset, 'par4_mediapipe_test2.pth')
    my_dataset = torch.load('morph_dataset/par5_mediapipe_test.pth')
    validation_loader = data.DataLoader(my_dataset, batch_size=data_config.BATCH_SIZE, shuffle=False, num_workers=8,
                                  pin_memory=True)
    return validation_loader

# Main training function
def main(config=None):
    # Start a wandb sweep
    with wandb.init(config=config):
        config = wandb.config

        # Load model
        model = modelSkeletonMorphing.Synthesizer(
            dropout_rate=config.dropout,
            num_blocks=config.num_blocks,
            layer_size=config.layer_size
        ).cuda()

        # Parameters for optimization
        params = list(model.parameters())

        # Loss function
        criterion = MPJPELoss()

        # Setting anomaly detection for autograd
        optimizer = optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay,
                                amsgrad=True, foreach=True)

        # Setting anomaly detection for autograd
        torch.autograd.set_detect_anomaly(True)

        # Load dataset
        train, eval, test, standardizer = load_train_test_all(config.datafolder, config.pars)
        train_loader = data.DataLoader(train, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
        eval_loader = data.DataLoader(eval, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
        test_loader = data.DataLoader(test, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

        # Start loss
        last_loss = 1000000

        # Training loop
        for epoch in range(config.epochs):
            # Training the model
            train_loss = NetworkTrainer.train(model, train_loader, optimizer, criterion)
            losses, pred_poses, pose_gt_batch, pose_inf_batch = NetworkTrainer.validation(model, test_loader, criterion)

            NetworkTrainer.log_training_result(train_loss, losses, pred_poses, pose_gt_batch, pose_inf_batch, epoch)

            # Print loss and epoch
            print('Finished epoch ' + str(epoch+1) + ' of ' + str(config.epochs) + ' with loss ' + str(np.sqrt(np.mean(losses))))

            # Save so far best model
            if np.mean(losses) < last_loss:
                model_path = 'models/morph_' + config.model_type + '_' + sweep_id + '.pth'
                torch.save(model.state_dict(), model_path)
                last_loss = np.mean(losses)

        # Save best model to wandb
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)


if __name__ == '__main__':
    datapath = "E:\MoCap"
    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=2)
    # Training complete
    print('done')
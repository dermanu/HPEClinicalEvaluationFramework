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
import skeletonMorphing.modelSkeletonMorphing as modelSkeletonMorphing
from types import SimpleNamespace
import torch.nn as nn
import wandb
import numpy as np
from utils.plotKeypoints import plot_3d_keypoints, plot_3d_keypoints_all
import tensorflow as tf

def print_all_data(datafolder: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i in range(11, 27):
        if i == 13:
            continue
        print(f'{data_folder}/par_{i}_mediapipe_dataset.pth')

        my_dataset1 = torch.load(f'{data_folder}/par_{i}_mediapipe_dataset.pth', map_location=torch.device(device))
        #my_dataset2 = torch.load('morph_dataset/par5_mediapipe_test.pth')
        #my_dataset2 = torch.load('morph_dataset/par5_mediapipe_test.pth')
        #my_dataset = torch.utils.data.ConcatDataset([my_dataset1, my_dataset2])


        train, test = my_dataset1.get_train_test()
        print(my_dataset1)
        print(train)
        print(test)

def train(datapath: str):
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

    mode = "disabled"
    # WandB – Initialize a new run
    wandb.init(project="skeleton-morphing", config=config, mode=mode)

    ## Remove all the depricated warnings caused by pip packages
    tf.get_logger().setLevel("ERROR")
    # Folder containing data
    #data_folder = '/home/emanu/Desktop/SegmentedData'
    data_folder = datapath + '/morph_dataset'

    # Parameters for training
    par = [5]
    mov = list(range(1, 18))
    cam = list(range(0, 6))
    model_type = 'mediapipe'
    num_cam = 6

    # Data file for training
    # config.datafile = data_folder + 'h36m_train_mpi_skeleton_pred.pickle'

    # Creating dataset and data loader
    #print('Creating dataset of participant' + str(par) + '...')
    #my_dataset = ReadDatasetFiles(data_folder, par, mov, cam, model_type)
    #torch.save(my_dataset, 'par4_mediapipe_test2.pth')
    #print('done')
    #print_all_data(data_folder)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    i = 12
    my_dataset1 = torch.load(f'{data_folder}/par_{i}_mediapipe_dataset.pth', map_location=torch.device(device))
    train, test = my_dataset1.get_train_test()


    train_loader = data.DataLoader(train, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = data.DataLoader(test, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    print('Data loader created')

    # Initializing the model (Synthesizer) and moving it to GPU
    model = modelSkeletonMorphing.Synthesizer().cuda()

    wandb.watch(model, log_freq=100)

    # Mean Squared Error Loss
    mse_loss = nn.MSELoss()

    # Number of epochs
    N_epochs = 1

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

    model.train()
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
            print(loss)

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_mean.append(loss.item())


        wandb.log({"loss": np.mean(losses_mean)})
        prediction = pred_poses.view(-1, pose_gt_batch.size(1), pose_gt_batch.size(2)).cpu().detach().numpy()[0]
        ground_truth = pose_gt_batch.cpu().detach().numpy()[0]
        hpe_truth = pose_inf_batch.cpu().detach().numpy()[0]
        plot_3d_keypoints(prediction, 'mediapipe', 'morphed', epoch)
        plot_3d_keypoints(hpe_truth, 'mediapipe', 'ground_truth', epoch)
        plot_3d_keypoints(hpe_truth, 'mediapipe', 'hpe_truth', epoch)
        plot_3d_keypoints_all(prediction, ground_truth, hpe_truth, 'mediapipe', epoch)
        wandb.log({"epoch": epoch})

        # Saving the model after each epoch
        print('Finished epoch ' + str(epoch) + ' of ' + str(N_epochs) + ' with loss ' + str(np.mean(losses_mean)))
        if np.mean(losses_mean) < last_loss_mean:
            last_loss_mean = np.mean(losses_mean)
            torch.save(model, f'models/model_skeleton_morph_par_{i}_mediapipe.pt')

        losses_mean = []

    mean_test_loss = []
    model.eval()
    for batch in test_loader:
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
        loss = mse_loss(pred_poses, output_poses)
        mean_test_loss.append(loss.detach().cpu().numpy())

    print(f"Test Loss", np.mean(mean_test_loss))

    # Training complete
    print('done')


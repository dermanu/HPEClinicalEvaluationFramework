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
from skeletonMorphing.loadMorphDatasets import list_to_file_name
import time
import os

class NetworkTrainer:

    @staticmethod
    def train_model(model, train_loader, optimizer,  criterion, epochs = 10, pars = np.arange(10, 27)):
        last_loss_mean = 100000
        model.train()
        # Training loop
        losses_mean = []
        for epoch in range(epochs):
            for batch in train_loader:
                # Access data for each batch
                pose_gt_batch = batch['pose_gt']
                pose_inf_batch = batch['pose_inf']

                # Creating tensors for input and output poses
                inp_poses = pose_inf_batch.view(-1, pose_inf_batch.size(1) * pose_inf_batch.size(2)).cuda().float()  # batches/frames x cams, keypoints x 3
                output_poses = pose_gt_batch.view(-1, pose_gt_batch.size(1) * pose_gt_batch.size(2)).cuda().float()

                # Forward pass through the model
                pred_poses = model(inp_poses)

                # Calculating MSE loss
                loss = criterion(pred_poses, output_poses)
                #print(loss)
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
            print('Finished epoch ' + str(epoch) + ' of ' + str(epochs) + ' with loss ' + str(np.mean(losses_mean)))
            if np.mean(losses_mean) < last_loss_mean:
                last_loss_mean = np.mean(losses_mean)
                i = list_to_file_name(pars)
                torch.save(model, f'models/trained/model_skeleton_morph_par_{i}_mediapipe.pt')

            losses_mean = []

    @staticmethod
    def test_model(model, test_loader, criterion):
        model.eval()
        mean_test_loss = []
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
            loss = criterion(pred_poses, output_poses)
            mean_test_loss.append(loss.detach().cpu().numpy())

        print(f"Test Loss", np.mean(mean_test_loss))

def load_train_test_all(data_folder: str, pars = np.arange(10, 27)):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = None
    test_dataset = None
    for i in pars:
        if i == 13:
            continue
        print(f'{data_folder}/par_{i}_mediapipe_dataset.pth')

        par_dataset = torch.load(f'{data_folder}/par_{i}_mediapipe_dataset.pth', map_location=torch.device(device))
        try:
            if  train_dataset is None:
                train_dataset, test_dataset = par_dataset.get_train_test()
            else:
                train, test = par_dataset.get_train_test()
                train_dataset = torch.utils.data.ConcatDataset([train_dataset, train])
                test_dataset = torch.utils.data.ConcatDataset([test_dataset, test])

        except Exception as e:
            print(e)

    return train_dataset, test_dataset

def train(datapath: str):
    # Configuration settings using SimpleNamespace
    config = SimpleNamespace()
    config.learning_rate = 0.0001
    config.BATCH_SIZE = 32
    config.N_epochs = 100
    config.log_interval = 100
    config.weight_decay = 1e-5
    mode = "online"

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

    ## Somethign wrong with 10 and 26
    pars = np.arange(10, 27)
    #Male: 12, 14
    # Female: 15, 16
    pars = np.array([12, 14, 15, 16])
    start_time = time.time()
    train, test = load_train_test_all(data_folder, pars)
    print('Data loaded in')
    print("--- %s seconds ---" % (time.time() - start_time))
    if not os.path.exists(data_folder + "/all_par_train.pth"):
        start_time = time.time()
        print("Saving Train")
        torch.save(train, data_folder + "/all_par_train.pth")
        print('Train saved in')
        print("--- %s seconds ---" % (time.time() - start_time))
    if not os.path.exists(data_folder + "/all_par_test.pth"):
        start_time = time.time()
        print("Saving Test")
        torch.save(test, data_folder + "/all_par_test.pth")
        print('Test saved in')
        print("--- %s seconds ---" % (time.time() - start_time))

    train_loader = data.DataLoader(train, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = data.DataLoader(test, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    print('Data loader created')
    # Initializing the model (Synthesizer) and moving it to GPU
    model = modelSkeletonMorphing.Synthesizer().cuda()

    wandb.watch(model, log_freq=100)

    # Mean Squared Error Loss
    mse_loss = nn.MSELoss()

    # Parameters for optimization
    params = list(model.parameters())  # + list(dec.parameters())

    # Setting anomaly detection for autograd
    optimizer = optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 95], gamma=0.1)

    # Setting anomaly detection for autograd
    torch.autograd.set_detect_anomaly(True)

    # Namespace to store losses during training

    NetworkTrainer.train_model(model = model,
                               train_loader = train_loader,
                               optimizer = optimizer,
                               criterion = mse_loss,
                               epochs = config.N_epochs,
                               pars = pars)


    NetworkTrainer.test_model(model = model, test_loader = test_loader, criterion = mse_loss)

    print('done')


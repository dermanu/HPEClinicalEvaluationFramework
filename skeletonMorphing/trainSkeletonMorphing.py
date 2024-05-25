"""
This script trainSkeletonMorphing trains the model modelSkeletonMorphing based on the ground throuth of two different
dataset.
The morphing network should be trained one subject of each dataset. Subject one then has to be excluded for all other
experiments to not include ground truth data.
It is based on the work of Bastian Wandt (https://github.com/bastianwandt/CanonPose/tree/main and
https://arxiv.org/pdf/2011.14679.pdf).
"""
import random

import torch.optim
from torch.utils import data
import torch.optim as optim
import tqdm
import pickle

import skeletonMorphing.modelSkeletonMorphing as modelSkeletonMorphing
from types import SimpleNamespace
import torch.nn as nn
import wandb
import numpy as np
from utils.plotKeypoints import plot_3d_keypoints, plot_3d_keypoints_all
import tensorflow as tf
from skeletonMorphing.loadMorphDatasets import list_to_file_name
from torch.utils.data import DataLoader, Dataset, Sampler
from utils.metrics import torch_calculate_mpjpe
import time
import os
from torch.utils.data import DataLoader, SubsetRandomSampler

class EveryNthSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, n):
        self.data_source = data_source
        self.n = n

    def __iter__(self):
        return iter(range(0, len(self.data_source), self.n))

    def __len__(self):
        return (len(self.data_source) + self.n - 1) // self.n

class Normalize():
    def __init__(self):
        """
        Initialize the dictionary with the std and mean of the training data
        """
        self.dict = {}

    def add_key(self, key, mins, maxs):
        """
        Add a key to the dictionary (training data)
        :param key:
        :param mean:
        :param std:
        :return:
        """
        if key in self.dict:
            c_mins, c_maxs = self.dict[key]
            if c_mins > mins:
                c_mins = mins
            if c_maxs < maxs:
                c_maxs = maxs

            self.dict[key] = (c_mins, c_maxs)
        else:
            self.dict[key] = (mins, maxs)

    def add_key_from_vector(self, vector, key):
        """
        Add a key to the dictionary (training data) using a vector
        :param vector:
        :param key:
        :return:
        """

        if vector.size == 0:
            return
        self.add_key(key, np.min(vector), np.max(vector))

    def scale(self, vector, key):
        """
        Standardize the vector using the mean and std from the dictionary
        :param vector:
        :param key:
        :return:
        """
        mins, maxs = self.dict[key]

        #print(mins, maxs)
        normalized = (vector - mins) / (maxs - mins)
        #print("Standardized value range: ", key, np.min(normalized), np.max(normalized))
        return normalized

    def descale(self, vector, key):
        """
        Destandardize the vector using the mean and std from the dictionary
        :param vector:
        :param key:
        :return:
        """
        return vector * (self.dict[key][1] - self.dict[key][0]) + self.dict[key][0]

    def save(self, path):
        """
        Save the dictionary to a file
        :param path:
        :return:
        """
        with open(path, 'wb') as f:
            pickle.dump(self.dict, f)

    @staticmethod
    def load(path):
        """
        Load the dictionary from a file
        :param path:
        :return:
        """
        normalize = Normalize()
        with open(path, 'rb') as f:
            normalize.dict = pickle.load(f)
        return normalize

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class MPJPELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, skeletons1, skeletons2):
        # Calculate MAE for Each Coordinate Error per Joint
        coordinate_abs_diff = torch.abs(skeletons1 - skeletons2)  # Absolute difference between corresponding coordinates
        mae_per_coordinate = torch.mean(coordinate_abs_diff, dim=0)  # MAE for each coordinate error per joint
        # Calculate MAE for Each Joint Position
        mae_per_joint_position = torch.mean(mae_per_coordinate, dim=1)  # MAE for each joint position
        # Calculate MAE for All Joint Positions
        overall_mae = torch.mean(mae_per_joint_position)  # Overall MAE across all joint positions
        return overall_mae

class NetworkTrainer:

    @staticmethod
    def validation(model, validation_loader, criterion, scaler, epoch = 0, debug = False, fold_id = -1):
        """
        Validation of the model
        :param criterion:
        :param model: Morphing model to train
        :param validation_loader: training data loader
        :return: Average loss of the model
        """
        # Iterate through batches
        model.eval()
        with torch.no_grad():
            losses = 0
            joints = []
            losses = []
            idx = 0
            pose = []
            for step, batch in enumerate(tqdm.tqdm(validation_loader, desc="Validation progress", leave=False)):
                # Access data for each batch
                pose_gt_batch = batch['pose_gt']
                pose_inf_batch = batch['pose_inf']
                par = batch['par']
                conf_inf = batch['confidences_inf'].cuda()

                # Creating tensors for input and output poses batches/frames x cams, keypoints x 3
                inp_poses = pose_inf_batch.view(-1, pose_inf_batch.size(1) * pose_inf_batch.size(2) * pose_inf_batch.size(3)).cuda().float().clone()
                output_poses = pose_gt_batch.view(-1,
                                                  pose_gt_batch.size(1) * pose_gt_batch.size(2)).cuda().float().clone()

                # Forward pass through the model
                pred_poses = model(inp_poses)


                for i, p in enumerate(par):
                    pred_poses[i] = scaler.descale(pred_poses[i], f"pose_gt_{p}")
                    output_poses[i] = scaler.descale(output_poses[i], f"pose_gt_{p}")
                    a = pred_poses.reshape(-1, 16, 3)
                    b = output_poses.reshape(-1, 16, 3)
                    joints = torch.abs(a - b)
                    wandb.log({f"loss_{p}": criterion(pred_poses[i], output_poses[i]).item(), "epoch": epoch + 1,
                               "fold_id": fold_id})
                    if debug:
                        #print("DEBUG")

                        column_mapping = {
                            'RShoulder': 'RSJC', # 12 - 0
                            'LShoulder': 'LSJC', # 11 - 1
                            'RElbow': 'REJC', # 14 - 2
                            'LElbow': 'LEJC', # 13 - 3
                            'RWrist': 'RWJC', # 16 - 4
                            'LWrist': 'LWJC', # 15 - 5
                            'RHip': 'RHJC', # 24 - 6
                            'LHip': 'LHJC', # 23 - 7
                            'RKnee': 'RKJC', # 26 - 8
                            'LKnee': 'LKJC', # 25  - 9
                            'RAnkle': 'RAJC', # 28 - 10
                            'LAnkle': 'LAJC', # 27 - 11
                            'RHeel': 'RHEE', # 30 - 12
                            'LHeel': 'LHEE', # 29 - 13
                            'RFootIndex': 'RTOE', # 32 - 14
                            'LFootIndex': 'LTOE', # 31 - 15
                        }
                        joint = joints[i].detach().cpu().numpy()
                        for y, k in enumerate(column_mapping.keys()):
                            wandb.log({f"{column_mapping[k]}_{p}": np.mean(joint[y]), "epoch": epoch + 1, "fold_id" : fold_id})

                # Calculating MSE loss
                loss = criterion(pred_poses, output_poses)
                if len(losses) != 0 and loss > losses[idx]:
                    idx = step

                # Log the loss of each batch
                wandb.log({"batch_loss": loss.item(), "batch": step + 1, "fold_id" : fold_id})
                losses.append(loss.detach().cpu().numpy().item())

                pose.append((pred_poses[0], pose_gt_batch[0], pose_inf_batch[0], par[0], conf_inf[0]))


        return losses, pose[idx][0], pose[idx][1], pose[idx][2], pose[idx][3], pose[idx][4]
        #return losses, pred_poses, pose_gt_batch, pose_inf_batch, par, np.mean(np.mean(joints, axis=0)), conf_inf
    

    @staticmethod
    def train(model, train_loader, optimizer, criterion, scaler):
        """
        Training the model
        :param criterion:
        :param model: Morphing model to train
        :param train_loader: training data loader
        :param optimizer: optimizer for the model
        :return: Average loss of the model
        """
        # Iterate through batches
        model.train()
        losses = []

        for step, batch in enumerate(tqdm.tqdm(train_loader, desc="Training progress", leave=False)):
            # Access data for each batch
            #print(batch)
            pose_gt_batch = batch['pose_gt'].clone()
            pose_inf_batch = batch['pose_inf'].clone()
            par = batch['par']
            #conf_inf = batch['confidences_inf'].cuda()

            inp_poses = pose_inf_batch.view(-1, pose_inf_batch.size(1) * pose_inf_batch.size(2) * pose_inf_batch.size(3)).cuda().float().clone()
            output_poses = pose_gt_batch.view(-1, pose_gt_batch.size(1) * pose_gt_batch.size(2)).cuda().float().clone()


            # Forward pass through the model
            pred_poses = model(inp_poses)
            #print(pred_poses)

            for i, p in enumerate(par):
                pred_poses[i] = scaler.descale(pred_poses[i], f"pose_gt_{p}")
                output_poses[i] = scaler.descale(output_poses[i], f"pose_gt_{p}")


            # Calculating MSE loss
            loss = criterion(pred_poses, output_poses)

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy().item())



        return losses

    @staticmethod
    def log_training_result(train_loss, losses, pred_poses, pose_gt_batch, pose_inf_batch, epoch, fold_id = -1):
        wandb.log({"train_loss": np.mean(train_loss), "validation_loss": np.mean(losses),
                   "validation_std" : np.std(losses), "rmse_loss" : np.sqrt(np.mean(losses)),
                   "epoch": epoch + 1, "fold_id": fold_id})


        prediction = pred_poses.view(pose_gt_batch.size(0), pose_gt_batch.size(1)).cpu().detach().numpy()
        ground_truth = pose_gt_batch.cpu().detach().numpy()
        hpe_truth = pose_inf_batch.cpu().detach().numpy()[0]
        #conf_inf = conf_inf.cpu().detach().numpy()
        print(pred_poses.shape)
        print(hpe_truth.shape)
        print(ground_truth.shape)
        
        plot_3d_keypoints(prediction, 'mediapipe', 'morphed', epoch, fold_id)
        plot_3d_keypoints(ground_truth, 'mediapipe', 'ground_truth', epoch, fold_id)
        plot_3d_keypoints(-hpe_truth, 'mediapipe', 'hpe_truth', epoch, fold_id)

        plot_3d_keypoints_all(prediction, ground_truth, -hpe_truth, 'mediapipe', epoch, fold_id)


    @staticmethod
    def train_model(model, train_loader, validation_loader, optimizer, criterion, epochs=10, pars=np.arange(10, 27),
                    config=None, scaler=None, debug = False, fold_id = -1):
        """
        Method to train model
        :param validation_loader:
        :param model:
        :param train_loader:
        :param optimizer:
        :param criterion:
        :param epochs:
        :param pars:
        :return:
        """
        last_loss_mean = 100000
        # Training loop
        for epoch in range(epochs):
            # time.sleep(15) #??
            train_loss = NetworkTrainer.train(model, train_loader, optimizer, criterion, scaler)
            losses, pred_poses, pose_gt_batch, pose_inf_batch, par,  conf_inf = NetworkTrainer.validation(model, validation_loader,
                                                                                          criterion, scaler, epoch, debug=debug, fold_id = fold_id)
            print('Finished epoch', epoch, 'of', epochs, 'with loss MSE:', np.mean(losses), ", ", np.std(losses))
            #print(losses)

            for i, p in enumerate([par]):
                pose_gt_batch = scaler.descale(pose_gt_batch, f"pose_gt_{p}")
                pose_inf_batch = scaler.descale(pose_inf_batch, f"pose_gt_{p}")


            NetworkTrainer.log_training_result(train_loss, losses, pred_poses, pose_gt_batch, pose_inf_batch, epoch, fold_id=fold_id)

            # Saving the model after each epoch

            if np.mean(losses) < last_loss_mean:
                last_loss_mean = np.mean(losses)
                i = list_to_file_name(pars)
                torch.save(model.state_dict(),
                           f'models/trained/model_skeleton_morph_mediapipe_id_{fold_id}_mediapipe_mpjpe.pth')


    @staticmethod
    def test_model(model, test_loader, criterion):
        """
        Method to test model
        :param model:
        :param test_loader:
        :param criterion:
        :return:
        """

        model.eval()
        mean_test_loss = []
        for batch in test_loader:
            # Access data for each batch
            pose_gt_batch = batch['pose_gt']
            pose_inf_batch = batch['pose_inf']

            # Creating tensors for input and output poses
            inp_poses = pose_inf_batch.view(-1, pose_inf_batch.size(1) * pose_inf_batch.size(2) * pose_inf_batch.size(3)).cuda().float().clone()  # batches/frames x cams, keypoints x 3
            output_poses = pose_gt_batch.view(-1, pose_gt_batch.size(1) * pose_gt_batch.size(2)).cuda().float()

            # Forward pass through the model
            pred_poses = model(inp_poses)

            # Calculating MSE loss
            loss = criterion(pred_poses, output_poses)
            mean_test_loss.append(loss.detach().cpu().numpy())

        print(f"Test Loss", np.sqrt(np.mean(mean_test_loss)))

def load_dataset_par(data_folder: str, par: int, scaler, concat = False):
    """
    Method to load training and test data for a participant
    :param data_folder:
    :param par:
    :return:
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    par_dataset = torch.load(f'{data_folder}/morph_dataset/par_{par}_mediapipe_dataset.pth',
                             map_location=torch.device(device))
    train_dataset, test_dataset = par_dataset.get_train_test()
    filter = False

    for d in train_dataset.datasets:
        if filter:
            d.filter_data()

        scaler.add_key_from_vector(d.csv_data, f"pose_gt_{par}")



    for d in test_dataset.datasets:
        if filter:
            d.filter_data()
        scaler.add_key_from_vector(d.csv_data, f"pose_gt_{par}")


    # scaler_test = scaler_train
    for _, d in enumerate(train_dataset.datasets):
        if d.csv_data.size == 0:
            continue

        train_dataset.datasets[_].csv_data = scaler.scale(d.csv_data,  f"pose_gt_{par}")
        train_dataset.datasets[_].par = par
        d.align_procrustes()

    for _, d in enumerate(test_dataset.datasets):
        if d.csv_data.size == 0:
            continue
        test_dataset.datasets[_].csv_data = scaler.scale(d.csv_data,  f"pose_gt_{par}")
        test_dataset.datasets[_].par = par
        d.align_procrustes()


    if concat:
        return torch.utils.data.ConcatDataset([train_dataset, test_dataset]), None

    return train_dataset, test_dataset

def load_train_test_all(data_folder: str, pars=np.arange(10, 27)):
    """
    Method to load all training and test data from participants [pars]
    :param data_folder:
    :param pars:
    :return:
    """

    scaler = Normalize()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dict = {}
    test_dict = {}
    filter = False
    concat = True
    for i in pars:
        if i == 13:
            continue

        train, test = load_dataset_par(data_folder, i, scaler, concat=concat)
        #print(f'{data_folder}/morph_dataset/par_{i}_mediapipe_dataset.pth')

        train_dict[i] = train
        if not concat:
            test_dict[i] = test


    #scaler.save("standardizer")
    print("Length of train dict", len(train_dict))
    print("Length of test dict", len(test_dict))
    return train_dict, test_dict, scaler 


def concat_dataset(dataset_dict: dict, pars=np.arange(10, 27)):
    dataset = None
    for i in pars:
        if i == 13:
            continue

        if dataset is None:
            dataset = dataset_dict[i]
        else:
            dataset = torch.utils.data.ConcatDataset([dataset, dataset_dict[i]])

    return dataset

def train_single_fold(config, datasets: tuple, scaler, missing_par: int, debug=False):
    """
    Note!!! Currently only using training data both for train and test since we moved to k-fold
    :param config:
    :param datasets:
    :param scaler:
    :param missing_par:
    :param debug:
    :return:
    """
    wandb.init(project="skeleton-morphing--moved", name=f'fold_{missing_par}', config=config, mode="online")

    train, test = datasets
    scaler = scaler
    sampler = EveryNthSampler(train, config.n_samples)
    shuffled_sampler = SubsetRandomSampler(list(sampler))
    sampler_test = EveryNthSampler(test, config.n_samples)
    shuffled_sampler_test = SubsetRandomSampler(list(sampler_test))

    wandb.log({"train_size": len(sampler), "test_size": len(sampler_test), "fold_id": missing_par})
    train_loader = data.DataLoader(train, batch_size=config.BATCH_SIZE, num_workers=8, pin_memory=True, sampler=shuffled_sampler)
    test_loader = data.DataLoader(test, batch_size=32, num_workers=8, pin_memory=True, sampler=shuffled_sampler_test)


    print('Data loader created')
    # Initializing the model (Synthesizer) and moving it to GPU
    model = modelSkeletonMorphing.Synthesizer().cuda()

    wandb.watch(model, log_freq=100)

    criterion = nn.MSELoss()

    # Parameters for optimization
    params = list(model.parameters())  # + list(dec.parameters())

    # Setting anomaly detection for autograd
    optimizer = optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 95], gamma=0.1)

    # Setting anomaly detection for autograd
    torch.autograd.set_detect_anomaly(True)

    # Namespace to store losses during training

    NetworkTrainer.train_model(model=model,
                               train_loader=train_loader,
                               validation_loader=test_loader,
                               optimizer=optimizer,
                               criterion=criterion,
                               epochs=config.N_epochs,
                               pars=config.pars,
                               config=config,
                               scaler=scaler,
                               debug = debug,
                               fold_id=missing_par)

    wandb.finish()




def train(datapath: str, pars, rand, mode, debug = False):
    # Configuration settings using SimpleNamespace
    # config = SimpleNamespace()
    # config.learning_rate = 0.0001
    # config.BATCH_SIZE = 32
    # config.N_epochs = 100
    # config.log_interval = 100
    # config.weight_decay = 1e-5
    # online/disabled for wandb
    mode = "online" if mode == True else "disabled"

    # Sweep configuration
    init_config = {
        'method': 'bayes',
        'metric': {
            'name': 'validation_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'value': 0.0001  # Learning rate: 1e-4 0.000
            },
            'BATCH_SIZE': {
                'value': 32  # Batch size: 32
            },
            'weight_decay': {
                'value': 1e-5  # Weight decay: 1e-5
            },
            'epochs': {
                'value': 100
            },
            'datafolder': {
                'value': datapath
            },
            'pars': {
                'value': pars
            },
            'model_type': {
                'value': 'mediapipe'
            },
            'N_epochs': {
                'value': 100
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 7,
            'eta': 3
        }
    }

    # WandB – Initialize a new run
    #wandb.init(project="skeleton-morphing--moved", config=init_config, mode=mode)
    # config = wandb.config['parameters']
    config = SimpleNamespace()
    config.learning_rate = 0.0001
    config.BATCH_SIZE = 32
    config.N_epochs = 100
    config.log_interval = 100
    config.weight_decay = 1e-5
    # config.pars = np.array([12])
    config.pars = pars
    config.model_type = "mediapipe"
    config.n_samples = 25
    print("PAR", pars)
    print("Estimating ", len(pars), "folds")
    train_dict, test_dict, scaler = load_train_test_all(datapath, pars)


    for par in pars:
        print("Fold with missing par: ", par)
        if par == 13:
            continue

        train = concat_dataset(train_dict, [y for y in pars if y != par])
        test = concat_dataset(train_dict, [par])

        train_single_fold(config, (train, test), scaler, par, debug)


    # Folder containing data
    # data_folder = '/home/emanu/Desktop/SegmentedData'
    print('done')

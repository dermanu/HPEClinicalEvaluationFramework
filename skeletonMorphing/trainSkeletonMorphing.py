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
        print("HERHEHRHEHREHERHEHRHREHRE",data_source)
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
    def validation(model, validation_loader, criterion, scaler):
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
                pred_poses = model(inp_poses, conf_inf)

                for i, p in enumerate(par):
                    pred_poses[i] = scaler.descale(pred_poses[i], f"pose_gt_{p}")
                    output_poses[i] = scaler.descale(output_poses[i], f"pose_gt_{p}")

                # print(pred_poses)
                a = pred_poses.reshape(-1, 16, 3)
                b = output_poses.reshape(-1, 16, 3)

                # joints.append(torch.mean(torch.abs(output_poses - pred_poses), axis=1).detach().cpu().numpy())

                for i in range(a.shape[0]):
                    joints.append(torch.mean(torch.abs(a[i] - b[i]), axis=1).detach().cpu().numpy())

                # Calculating MSE loss
                loss = criterion(pred_poses, output_poses)
                # Log the loss of each batch
                wandb.log({"batch_loss": loss.item(), "batch": step + 1})
                losses.append(loss.detach().cpu().numpy().item())

            joints = np.array(joints)
            #print("Joints", np.mean(joints, axis=1))
            #print("Joints MAE", np.mean(np.mean(joints, axis=0)))
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
            for i, k in enumerate(column_mapping.keys()):
                wandb.log({f"{column_mapping[k]}": np.mean(joints, axis=1)[i]})

        return losses, pred_poses, pose_gt_batch, pose_inf_batch, par, np.mean(np.mean(joints, axis=0))

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
            pose_gt_batch = batch['pose_gt']
            pose_inf_batch = batch['pose_inf']
            par = batch['par']
            conf_inf = batch['confidences_inf'].cuda()

            # Creating tensors for input and output poses batches/frames x cams, keypoints x 3
            inp_poses = pose_inf_batch.view(-1, pose_inf_batch.size(1) * pose_inf_batch.size(2) * pose_inf_batch.size(3)).cuda().float().clone()
            output_poses = pose_gt_batch.view(-1, pose_gt_batch.size(1) * pose_gt_batch.size(2)).cuda().float().clone()


            # Forward pass through the model
            pred_poses = model(inp_poses,conf_inf)
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
    def log_training_result(train_loss, losses, pred_poses, pose_gt_batch, pose_inf_batch, epoch, mpjpe_loss):
        wandb.log({"train_loss": np.mean(train_loss), "validation_loss": np.mean(losses),
                   "validation_std" : np.std(losses), "mpjpe_loss" : mpjpe_loss, "rmse_loss" : np.sqrt(np.mean(losses)),
                   "epoch": epoch + 1})
        idx = random.randint(0,pose_gt_batch.size(0)-1)

        prediction = pred_poses.view(-1, pose_gt_batch.size(1), pose_gt_batch.size(2)).cpu().detach().numpy()[idx]
        ground_truth = pose_gt_batch.cpu().detach().numpy()[idx]
        print(pose_inf_batch.cpu().detach().numpy()[idx].shape)
        print(np.mean(pose_inf_batch.cpu().detach().numpy()[idx], axis=0).shape)
        print(np.mean(pose_inf_batch.cpu().detach().numpy()[idx], axis=1).shape)

        hpe_truth = np.mean(pose_inf_batch.cpu().detach().numpy()[idx], axis=0)[:, 0:3]
        plot_3d_keypoints(prediction, 'mediapipe', 'morphed', epoch)
        plot_3d_keypoints(ground_truth, 'mediapipe', 'ground_truth', epoch)
        plot_3d_keypoints(hpe_truth, 'mediapipe', 'hpe_truth', epoch)

        plot_3d_keypoints_all(prediction, ground_truth, -hpe_truth, 'mediapipe', epoch)


    @staticmethod
    def train_model(model, train_loader, validation_loader, optimizer, criterion, epochs=10, pars=np.arange(10, 27),
                    config=None, scaler=None):
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
        scaler_train, scaler_test = scaler
        last_loss_mean = 100000
        # Training loop
        for epoch in range(epochs):
            # time.sleep(15) #??
            train_loss = NetworkTrainer.train(model, train_loader, optimizer, criterion, scaler_train)
            losses, pred_poses, pose_gt_batch, pose_inf_batch, par, mpjpe_loss = NetworkTrainer.validation(model, validation_loader,
                                                                                          criterion, scaler_test)
            print('Finished epoch', epoch, 'of', epochs, 'with loss MSE:', np.mean(losses), ", ", np.std(losses),
                  "MPJPE: ", mpjpe_loss)
            print(losses)

            for i, p in enumerate(par):
                pose_gt_batch[i] = scaler_test.descale(pose_gt_batch[i], f"pose_gt_{p}")
                pose_inf_batch[i] = scaler_test.descale(pose_inf_batch[i], f"pose_gt_{p}")


            NetworkTrainer.log_training_result(train_loss, losses, pred_poses, pose_gt_batch, pose_inf_batch, epoch, mpjpe_loss)

            # wandb.log({"epoch": epoch})
            #print('Finished epoch ' + str(epoch) + ' of ' + str(epochs) + ' with loss ' + str(np.mean(losses)))

            # Saving the model after each epoch

            if np.mean(losses) < last_loss_mean:
                last_loss_mean = np.mean(losses)
                i = list_to_file_name(pars)
                torch.save(model.state_dict(),
                           f'models/trained/model_skeleton_morph_{config.model_type}_par_{i}_mediapipe_mpjpe.pth')

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


def load_train_test_all(data_folder: str, pars=np.arange(10, 27)):
    """
    Method to load all training and test data from participants [pars]
    :param data_folder:
    :param pars:
    :return:
    """

    scaler_train = Normalize()
    #scaler_test = Normalize()
    scaler_test = scaler_train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = None
    test_dataset = None
    for i in pars:
        if i == 13:
            continue
        print(f'{data_folder}/morph_dataset/par_{i}_mediapipe_dataset.pth')


        par_dataset = torch.load(f'{data_folder}/morph_dataset/par_{i}_mediapipe_dataset.pth',
                                 map_location=torch.device(device))

        if train_dataset is None:
            train_dataset, test_dataset = par_dataset.get_train_test()
            print(train_dataset.datasets[0].csv_data.shape)
            print(train_dataset.datasets[0].pose_inf.shape)
            for d in train_dataset.datasets:

                scaler_train.add_key_from_vector(d.csv_data, f"pose_gt_{i}")
                #scaler_train.add_key_from_vector(i.pose_inf, "pose_inf")

            print(test_dataset)
            for d in test_dataset.datasets:
                scaler_test.add_key_from_vector(d.csv_data, f"pose_gt_{i}")
                #scaler_test.add_key_from_vector(i.pose_inf, "pose_inf")

            # scaler_test = scaler_train
            for _, d in enumerate(train_dataset.datasets):
                if d.csv_data.size == 0:
                    continue

                train_dataset.datasets[_].csv_data = scaler_train.scale(d.csv_data,  f"pose_gt_{i}")
                train_dataset.datasets[_].par = i

            for _, d in enumerate(test_dataset.datasets):
                if d.csv_data.size == 0:
                    continue
                test_dataset.datasets[_].csv_data = scaler_test.scale(d.csv_data,  f"pose_gt_{i}")
                test_dataset.datasets[_].par = i


        else:
            train, test = par_dataset.get_train_test()
            for d in train.datasets:
                scaler_train.add_key_from_vector(d.csv_data, f"pose_gt_{i}")
                #scaler_train.add_key_from_vector(i.pose_inf, "pose_inf")

            for d in test.datasets:
                scaler_test.add_key_from_vector(d.csv_data, f"pose_gt_{i}")

            for _, d in enumerate(train.datasets):
                if d.csv_data.size == 0:
                    continue

                train.datasets[_].csv_data = scaler_train.scale(d.csv_data,  f"pose_gt_{i}")
                train.datasets[_].par = i

            for _, d in enumerate(test.datasets):
                if d.csv_data.size == 0:
                    continue
                test.datasets[_].csv_data = scaler_test.scale(d.csv_data,  f"pose_gt_{i}")
                test.datasets[_].par = i

            train_dataset = torch.utils.data.ConcatDataset([train_dataset, train])
            test_dataset = torch.utils.data.ConcatDataset([test_dataset, test])

        print("NORM RESULTS")
        print(scaler_train.dict[f'pose_gt_{i}'])
        print(scaler_test.dict[f'pose_gt_{i}'])


    print(len(train_dataset))

    #scaler.save("standardizer")
    return train_dataset, test_dataset, scaler_train, scaler_test

def load_train_test_2(data_folder: str, pars = np.arange(10, 27)):
    """
    Method to load all training and test data from participants [pars]
    :param data_folder:
    :param pars:
    :return:
    """
    import random
    scaler_train = Normalize()
    scaler_test = scaler_train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = []
    test_dataset = []

    random.seed(42)

    for i in pars:
        if i == 13:
            continue

        par_dataset = torch.load(f'{data_folder}/morph_dataset/par_{i}_mediapipe_dataset.pth',
                                 map_location=torch.device(device))

        indexes = list(range(len(par_dataset)))
        random.shuffle(indexes)
        train_frames = indexes[:int(0.8 * len(indexes))]
        test_frames = indexes[int(0.8 * len(indexes)):]
        par_dataset.train_frames = train_frames
        par_dataset.test_frames = test_frames

        train_data, test_data = par_dataset.get_train_test()

        for d in train_data.datasets:
            scaler_train.add_key_from_vector(d.csv_data, f"pose_gt_{i}")
        for d in test_data.datasets:
            scaler_test.add_key_from_vector(d.csv_data, f"pose_gt_{i}")

        for d in train_data.datasets:
            d.csv_data = scaler_train.scale(d.csv_data, f"pose_gt_{i}")
            d.par = i
        for d in test_data.datasets:
            d.csv_data = scaler_test.scale(d.csv_data, f"pose_gt_{i}")
            d.par = i

        train_dataset.extend(train_data)
        test_dataset.extend(test_data)

    return train_dataset, test_dataset, scaler_train, scaler_test

def train(datapath: str, pars, rand, mode):
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
    wandb.init(project="skeleton-morphing", config=init_config, mode=mode)
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
    print("PAR", pars)

    # Folder containing data
    # data_folder = '/home/emanu/Desktop/SegmentedData'
    data_folder = datapath + '/morph_dataset'

    start_time = time.time()
    if rand:
        train, test, scaler_train, scaler_test = load_train_test_2(datapath, config.pars)
    else:
        train, test, scaler_train, scaler_test = load_train_test_all(datapath, config.pars)


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

    n = 5  # For picking every 5th sample
    sampler = EveryNthSampler(train, n)
    # Create a SubsetRandomSampler to shuffle the indices generated by EveryNthSampler
    #print("HERE GAIAN", len(sampler))
    shuffled_sampler = SubsetRandomSampler(list(sampler))
    #print("HERE GAIAN", len(shuffled_sampler))
    #n = 25  # For picking every 5th sample
    sampler_test = EveryNthSampler(test, n)
    # Create a SubsetRandomSampler to shuffle the indices generated by EveryNthSampler
    shuffled_sampler_test = SubsetRandomSampler(list(sampler_test))
    wandb.log({"train_size": len(sampler), "test_size": len(sampler_test)})
    train_loader = data.DataLoader(train, batch_size=config.BATCH_SIZE, num_workers=8, pin_memory=True, sampler=shuffled_sampler)
    print(train_loader)
    test_loader = data.DataLoader(test, batch_size=config.BATCH_SIZE, num_workers=8, pin_memory=True, sampler=shuffled_sampler_test)


    print('Data loader created')
    # Initializing the model (Synthesizer) and moving it to GPU
    model = modelSkeletonMorphing.Synthesizer().cuda()

    wandb.watch(model, log_freq=100)

    # Mean Squared Error Loss
    # mse_loss = nn.MSELoss()
    criterion = MPJPELoss()
    criterion = RMSELoss()
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
                               scaler=(scaler_train, scaler_test))

    print('done')

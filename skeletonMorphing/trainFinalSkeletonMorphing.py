"""
Training the morphing model on one fixed set of hyperparameters defined in configFinal.yaml. Optimized code to run on a
HPC.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F
import wandb
import numpy as np
import skeletonMorphing.modelSkeletonMorphing as modelSkeletonMorphing
from utils.plotKeypoints import plot_3d_keypoints_all
from itertools import islice
import torch.cuda.amp as amp
import torch.profiler
import yaml

# Joint names used for training
joint_names = [
    'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow', 'right_wrist', 'left_wrist', 'right_hip',
    'left_hip', 'right_knee', 'left_knee', 'right_ankle', 'left_ankle', 'right_heel', 'left_heel', 'right_foot_index',
    'left_foot_index'
]

# Set seeds for random number generator
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def group_by_n(participants, n):
    '''
    Helper function to split participants into folds for cross-validation

    Parameters:
    - participants: particpant indcies
    - n: fold size

    Returns:
    - Folds for leave-n-participants-out cross-validation
    '''

    it = iter(participants)
    return iter(lambda: list(islice(it, n)), [])


class EveryNthSampler(torch.utils.data.Sampler):
    '''
    Helper function that only uses every n-th data sample of the dataset to sort out similar poses that are temporarly
    close.
    '''
    def __init__(self, data_source, n):
        self.data_source = data_source
        self.n = int(n)

    def __iter__(self):
        return iter(range(0, len(self.data_source), self.n))

    def __len__(self):
        return (len(self.data_source) + self.n - 1) // self.n


def calculate_mpjpe(predicted_keypoints, target_keypoints):
    """
    Calculate the Mean Per Joint Position Error (MPJPE).

    Parameter:
    - predicted_keypoints (torch.Tensor): The predicted keypoints tensor of shape (batch_size, num_joints, 3).
    - target_keypoints (torch.Tensor): The ground truth keypoints tensor of shape (batch_size, num_joints, 3).

    Returns:
        float: The MPJPE value.
    """
    # Calculate Euclidean distance for each joint
    distances = torch.norm(predicted_keypoints - target_keypoints, dim=-1)

    # Average over all joints and all samples
    mpjpe = distances.mean().item()
    return mpjpe


# Class for network training
class NetworkTrainer:
    '''
    All functions needed for training, validating and testing the morphing model, as well as logging the results to
    wandb.
    '''

    @staticmethod
    def train(model, train_loader, optimizer, criterion, scaler):
        '''
        Training the morphing model

        Parameters:
        - model: Model to train
        - train_loader: Dataloader for the training data
        - optimizer: Optimizer
        - criterion: Loss Criterion
        - scaler: Uses mixed-precision training (amp.GradScaler) for efficiency.
        '''

        # Initialize model for training
        model.train()

        # Initialize variables
        losses = []

        # Iterates over the training data loader.
        for step, batch in enumerate(train_loader):
            pose_gt_batch = batch['pose_gt'].float().cuda() # Get ground truth (Qualisys)
            pose_inf_batch = batch['pose_inf'].float().cuda() # Get inferred data (HPE)
            conf_inf_batch = batch['confidences_inf'].float().cuda() # Get HPE prediction confidence score

            # Identify the camera with the highest confidence for each sample
            best_camera = conf_inf_batch.mean(dim=2).argmax(dim=1)
            batch_indices = torch.arange(pose_inf_batch.size(0), device=pose_inf_batch.device)
            pose_inf_batch = pose_inf_batch[batch_indices, best_camera]

            inp_poses = pose_inf_batch.view(-1, pose_inf_batch.size(1) * pose_inf_batch.size(2))
            output_poses = pose_gt_batch.view(-1, pose_gt_batch.size(1) * pose_gt_batch.size(2))

            if inp_poses is None:
                print(f'Skipping training batch {step} due to dimension mismatch')
                continue
            else:
                # Performs a forward pass and computes the loss
                try:
                    pred_poses = model(inp_poses)
                except Exception as e:
                    print(f"Error during model forward pass: {e}")
                    print(f"Input shape: {inp_poses.shape}")
                    raise
                loss = criterion(pred_poses, output_poses)

            # Backpropagation with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.detach().cpu().numpy().item())

        return losses


    @staticmethod
    def test(model, test_loader, criterion):
        '''
        Testing the trained morphing model

        Parameters:
        - model: Model to train
        - test_loader: Dataloader for the test data
        - criterion: Loss Criterion
        '''
        # Initialize model for testing
        model.eval()

        # Start testing
        with (((torch.no_grad()))):
            # Initialize variables
            losses = []
            mpjpe_without_values = []
            mpjpe_values = []
            mpjpe_joint_values = []
            predictions = []
            ground_truths = []
            hpe_truths = []

            # Go through all batches of the testing dataset
            for step, batch in enumerate(test_loader):
                pose_gt_batch = batch['pose_gt'].float().cuda()
                pose_inf_batch = batch['pose_inf'].float().cuda()
                conf_inf_batch = batch['confidences_inf'].float().cuda() # Get HPE prediction confidence score

                # Randomly select one camera view
                #num_cameras = pose_inf_batch.size(1)
                #random_camera = random.randint(0, num_cameras - 1)
                #pose_inf_batch = pose_inf_batch[:, random_camera, :, :]

                # Identify the camera with the highest confidence for each sample, only if the highest confidence > 0.6
                best_camera = conf_inf_batch.mean(dim=2).argmax(dim=1)
                batch_indices = torch.arange(pose_inf_batch.size(0), device=pose_inf_batch.device)
                pose_inf_batch = pose_inf_batch[batch_indices, best_camera]

                inp_poses = pose_inf_batch.view(-1, pose_inf_batch.size(1) * pose_inf_batch.size(2))
                output_poses = pose_gt_batch.view(-1, pose_gt_batch.size(1) * pose_gt_batch.size(2))

                # Check if input and output have the same size (and are available). Otherwise, skip.
                if inp_poses is None:
                    print(f'Skipping testing batch {step} due to dimension mismatch')
                    continue

                # Forward pass
                pred_poses = model(inp_poses)

                # Calculate the difference-based loss
                loss = criterion(pred_poses, output_poses)
                losses.append(loss.detach().cpu().numpy().item())

                # Reshape predictions and ground truth for MPJPE calculation
                pred_poses_reshaped = pred_poses.view(-1, 16, 3)  # Assuming 16 keypoints
                output_poses_reshaped = output_poses.view(-1, 16, 3)
                input_poses_reshaped = inp_poses.view(-1, 16, 3)

                # Calculate MPJPE for the batch
                mpjpe = calculate_mpjpe(pred_poses_reshaped, output_poses_reshaped)
                mpjpe_without = calculate_mpjpe(output_poses_reshaped, input_poses_reshaped)
                mpjpe_joint = {}
                for idx, joint_name in enumerate(joint_names):
                    mpjpe_joint[joint_name] = calculate_mpjpe((output_poses_reshaped[:,idx,:], input_poses_reshaped[:,idx,:])) - calculate_mpjpe((pred_poses_reshaped[:,idx,:], output_poses_reshaped[:,idx,:]))
                mpjpe_values.append(mpjpe)
                mpjpe_without_values.append(mpjpe_without)
                mpjpe_joint_values.append(mpjpe_joint)

                # Store predictions and ground truth for further analysis or logging
                predictions.append(pred_poses.cpu())
                ground_truths.append(output_poses.cpu())
                hpe_truths.append(inp_poses.cpu())

        # Log data to wandb
        wandb.log({
            "test_loss": np.mean(losses),
             "test_mpjpe": np.mean(mpjpe_values),
             "test_mpjpe_without": np.mean(mpjpe_without_values),
        })
        for joint in joint_names:
            wandb.log({f"test_{joint}": np.mean([mpjpe[joint] for mpjpe in mpjpe_joint_values])})

        # Optionally, plot and log some predictions
        if len(predictions) > 0:
            prediction = predictions[0].view(-1, 3).cpu().detach().numpy()  # Reshape to (num_keypoints, 3)
            ground_truth = ground_truths[0].view(-1, 3).cpu().detach().numpy()
            hpe_truth = hpe_truths[0].view(-1, 3).cpu().detach().numpy()

            plot_3d_keypoints_all(prediction[1], ground_truth[1], hpe_truth[1], 'mediapipe', 0)
            plot_3d_keypoints_all(prediction[100], ground_truth[100], hpe_truth[100], 'mediapipe', 0)
            plot_3d_keypoints_all(prediction[200], ground_truth[200], hpe_truth[300], 'mediapipe', 0)
            plot_3d_keypoints_all(prediction[300], ground_truth[300], hpe_truth[300], 'mediapipe', 0)


        return losses, mpjpe_values, mpjpe_without_values, predictions, ground_truths


    @staticmethod
    def train_model(data_dict, test_dict, epochs=50, pars=None, pars_test=None, config=None):
        '''
        Trains and test the model across k-folds and logs the results to wandb.

        Parameters:
        - data_dict: Dictionary of training and evaluation data.
        - test_dict: Dictionary of test data.
        - epochs: Number of epochs per fold.
        - pars: List of participants for training and evaluation
        - pars_test: List of participants for testing
        - config: Configuration dictionary with training parameters.
        '''

        # Final training on all data if required
        print('Training final model on all data.')

        # Re-initialize model, optimizer, scheduler, and scaler
        model = modelSkeletonMorphing.Synthesizer(config.dropout_rate, config.layer_size).cuda()
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        scaler = amp.GradScaler()

        full_data = concat_dataset(data_dict, pars)
        full_data_loader = DataLoader(full_data, batch_size=config.BATCH_SIZE, num_workers=12, pin_memory=True)

        for epoch in range(epochs):
            print(f"Epoch [{epoch + 1}/{epochs}]")
            final_train_loss = NetworkTrainer.train(model, full_data_loader, optimizer, criterion, scaler)
            avg_train_loss = np.mean(final_train_loss)
            scheduler.step(avg_train_loss)

        # Testing the final model
        test_data = concat_dataset(test_dict, pars_test)
        test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, num_workers=12, pin_memory=True)
        test_losses, mpjpe, mpjpe_without, predictions, ground_truths = NetworkTrainer.test(model, test_loader,
                                                                                            criterion)

        # Save the final model weights
        torch.save(model, 'models/trained/model_skeleton_morph_mediapipe_final.pth')
        print("Final model saved.")


# Load dataset for a specific participant
def load_dataset_par(data_folder: str, par: int):
    '''
    Loads and preprocesses the dataset for a specific participant.

    Parameters:
    - data_folder: Path to the folder where the dataset is located.
    - par: Participant number.

    Returns:
    - Torch dataset for training and evaluation.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    par_path = f'{data_folder}/morph_dataset/par_{par}_mediapipe_dataset.pth'

    # Load dataset
    try:
        par_dataset = torch.load(par_path, map_location=device, weights_only=False)
        train_dataset, eval_dataset = par_dataset.get_train_test()
    except FileNotFoundError:
        print(f"Error: Dataset for participant {par} not found at {par_path}.")
        return None, None
    except Exception as e:
        print(f"Error loading dataset for participant {par}: {e}")
        return None, None

    # Check datasets
    if len(train_dataset.datasets) == 0:
        print(f"Warning: Empty training dataset for participant {par}.")
    if len(eval_dataset.datasets) == 0:
        print(f"Warning: Empty test dataset for participant {par}.")

    # Align keypoints of dataset using Procrustes alignment
    for idx, d in enumerate(train_dataset.datasets):
        if d.csv_data.size == 0:
            continue
        train_dataset.datasets[idx].par = par
        d.align_procrustes()

    for idx, d in enumerate(eval_dataset.datasets):
        if d.csv_data.size == 0:
            continue
        eval_dataset.datasets[idx].par = par
        d.align_procrustes()

    return torch.utils.data.ConcatDataset([train_dataset, eval_dataset])


def load_train_test_all(data_folder: str, train_eval_pars, test_pars):
    """
    Load all training and test data for participants

    Parameters:
    - data_folder: Path to the folder where the dataset is located.
    - train_eval_pars: List of participants for training and evaluation
    - test_pars: List of participants for testing

    Returns:
    - train_eval_dict: Dictionary of training and evaluation data.
    - test_dict: Dictionary of test data.
    """
    train_eval_dict = {}
    for i in train_eval_pars:
        train_eval= load_dataset_par(data_folder, i)
        train_eval_dict[i] = train_eval

    test_dict = {}
    for i in test_pars:
        test = load_dataset_par(data_folder, i)
        test_dict[i] = test

    print("Length of train dict", len(train_eval_dict))
    print("Length of test dict", len(test_dict))

    return train_eval_dict, test_dict


def concat_dataset(dataset_dict: dict, pars=np.arange(4, 27)):
    '''
    Concatenates datasets for specified participants, excluding a participant for testing.

    Parameters:
    - dataset_dict: Dictionary of training and evaluation data.
    - pars: List of participants for training and evaluation

    Returns:
    - Combined dataset
    '''
    dataset = None
    for i in pars:
        if i not in dataset_dict:
            print(f"Warning: Participant {i} not found in dataset_dict.")
            continue
        if dataset_dict[i] is None:
            print(f"Warning: No data found for participant {i}.")
            continue
        dataset = dataset if dataset is not None else dataset_dict[i]
        dataset = torch.utils.data.ConcatDataset([dataset, dataset_dict[i]]) if dataset is not None else dataset_dict[i]
    return dataset if dataset is not None else []

@staticmethod
def load_and_test(model_path, test_loader, criterion):
    """
    Loads a pre-trained model and tests it on a provided dataset.

    Args:
        model_path (str): Path to the saved model file.
        test_loader (DataLoader): DataLoader for the test data.
        criterion (nn.Module): Loss function for evaluation.

    Returns:
        None
    """
    # Load the model
    print(f"Loading model from {model_path}")
    model = torch.load(model_path).cuda()

    # Test the loaded model
    losses, mpjpe_values, mpjpe_without_values, predictions, ground_truths = NetworkTrainer.test(
        model=model,
        test_loader=test_loader,
        criterion=criterion
    )

    print(f"Test completed. Average MPJPE: {np.mean(mpjpe_values):.4f}, Loss: {np.mean(losses):.4f}")


def sweep():
    """
    Initialize the hyperparameter sweep. Here just on set of parameters is given, thus no real sweep is done, but data
    is logged in Wandb.
    """
    # Get hyperparameters
    with open("skeletonMorphing/configFinal.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    run = wandb.init(config=config)

    # Define the data path
    datapath = r"C:\Users\vizlab_stud\emanuel"

    # Load the data
    train_eval_dict, test_dict = load_train_test_all(datapath, np.array(wandb.config.train_pars), np.array(wandb.config.test_pars))

    NetworkTrainer.train_model(
        data_dict=train_eval_dict,
        test_dict=test_dict,
        epochs=wandb.config.N_epochs,
        pars=wandb.config.train_pars,
        pars_test=wandb.config.test_pars,
        config=wandb.config
    )

    wandb.finish()

# if __name__ == "__main__":
#     sweep()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skeleton Morphing Training and Testing")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Mode to run the script: 'train' or 'test'.")
    parser.add_argument("--model_path", type=str, default="models/trained/model_skeleton_morph_mediapipe_final.pth",
                        help="Path to the saved model for testing.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset folder.")
    parser.add_argument("--test_pars", type=int, nargs="+", default=None,
                        help="Participant IDs for testing.")
    args = parser.parse_args()

    if args.mode == "train":
        # Training workflow
        with open("skeletonMorphing/configFinal.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        run = wandb.init(config=config)

        # Load the data
        train_eval_dict, test_dict = load_train_test_all(args.data_path, np.array(config.train_pars), np.array(config.test_pars))

        NetworkTrainer.train_model(
            data_dict=train_eval_dict,
            test_dict=test_dict,
            epochs=config.N_epochs,
            pars=config.train_pars,
            pars_test=config.test_pars,
            config=config
        )
        wandb.finish()

    elif args.mode == "test":
        # Testing workflow
        print("Loading test data...")
        _, test_dict = load_train_test_all(args.data_path, [], np.array(args.test_pars))

        test_data = concat_dataset(test_dict, pars=args.test_pars)
        test_loader = DataLoader(test_data, batch_size=64, num_workers=12, pin_memory=True)

        # Define criterion
        criterion = nn.MSELoss()

        # Load and test the model
        NetworkTrainer.load_and_test(
            model_path=args.model_path,
            test_loader=test_loader,
            criterion=criterion
        )

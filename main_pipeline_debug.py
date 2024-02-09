import os
import wandb
import numpy as np
import cv2
import torch
from skeletonMorphing import modelSkeletonMorphing
from utils import cameraCalibration as camCali
from utils import frameAugmentation as frameAug
from utils import metrics, postprocessing
from models import mediapipeMono


def log_frame_example(frames):
    """
    Log the last frame of one camera angle to visualize applied frame augmentations
    :param frames: augmented frames, e.g. [cam0, cam1, cam2, ...]
    :return:
    """
    image = wandb.Image(frames[0], caption=f"Frame example of current evaluation")
    wandb.log({"Example_Frame": image})


class Framework:
    def __init__(self, model_name, model_type, directory):
        # Access the parsed arguments
        self.model_name = model_name
        self.model_type = model_type
        self.dataset_path = directory

        # Initialize some functions
        self.model_skel_morph = self.load_morph_model()
        self.cam_desynchronizer = frameAug.CameraDesynchronizer()

        # Initialize empty global variables
        self.sweep_config = None

        # Participants of dataset used in pipeline
        self.participants = ['par5', 'par6', 'par7', 'par8', 'par9', 'par10', 'par11', 'par12', 'par14', 'par15',
                             'par16', 'par17', 'par18', 'par19', 'par20', 'par21', 'par22', 'par23', 'par24', 'par25',
                             'par26']

        # Defines movement number in dataset related to different movement categories
        self.movement_category = {
            "upper": [1, 2, 3, 4],
            "lower": [5, 6, 7, 8],
            "complex": [9, 10, 11, 12, 13],
            "sitting": [14, 15, 16, 17]
        }

        # Defines body segments
        self.body_segments = {
            "left_lower_arm": [16, 14],
            "left_upper_arm": [14, 12],
            "right_lower_arm": [15, 13],
            "right_upper_arm": [13, 11],
            "torso": [11, 12, 23, 24],
            "left_upper_leg": [24, 26],
            "left_lower_leg": [32, 30, 28, 26],
            "right_upper_leg": [23, 25],
            "right_lower_leg": [31, 29, 27, 25]}

        # Defines segments around each joint to calculate angles (distal to proximal).
        self.joint_segments = {
            "right_elbow": [15, 13, 11],
            "left_elbow": [16, 14, 12],
            "right_shoulder_1": [13, 11, 12],
            "right_shoulder_2": [13, 11, 23],
            "left_shoulder_1": [14, 12, 11],
            "left_shoulder_2": [14, 12, 24],
            # "torso": [11, 12, 23, 24],
            "right_hips_1": [25, 23, 24],
            "right_hips_2": [25, 23, 11],
            "left_hips_1": [26, 24, 23],
            "left_hips_2": [26, 24, 12],
            "right_knee": [23, 25, 27],
            "left_knee": [24, 26, 28],
            "right_ankle": [31, 27, 25],
            "left_ankle": [32, 28, 26]
        }

        self.interpolation_fun = "akima"
        self.smoothing_fun = "median"

    def load_data_paths(self, config):
        """
        Loads data paths of the Vizlab dataset. Finds paths of the video and respective csv movement files.
        :return:
        video_paths_all = [[cam0, cam1, cam2, ...], [cam1, cam2, cam3, ...], ...]
        csv_paths_all = [[cam0, cam1, cam2, ...], [cam1, cam2, cam3, ...], ...]
        """

        # Get relevant parameters from sweep configuration
        movement_numbers = self.movement_category[config['movement']]
        participants = self.sweep_config['dataset']
        cams = config['cameras']
        if isinstance(cams, int):
            string_cams = [str(cams)]  # Convert the single integer to a list containing its string representation
        elif isinstance(cams, list):
            string_cams = list(map(str, cams))

        # Loop through all the video and csv files and extract the for the sweep relevant ones
        video_paths_all = []
        csv_paths_all = []

        for participant in participants:  # Loop through participants
            folder_path = os.path.join(self.dataset_path, participant)
            if os.path.isdir(folder_path):  # Check if it's a directory
                for movement_number in movement_numbers:  # Loop for each chosen movement type
                    video_paths = []  # Reset single iteration arrays
                    csv_paths = []
                    for cam in string_cams:  # Loop for each camera
                        video_file_name = f"{participant}_Mov{movement_number}_Cam{cam}.avi"
                        csv_file_name = f"{participant}_Mov{movement_number}_Cam{cam}.csv"
                        video_file_name = os.path.join(folder_path, video_file_name)
                        csv_file_name = os.path.join(folder_path, csv_file_name)
                        # Check if the file names exist
                        if os.path.isfile(video_file_name):
                            video_paths.append(video_file_name)
                        if os.path.isfile(csv_file_name):
                            csv_paths.append(csv_file_name)

                    # Save all paths of all cameras for each movement iteration. Check if all required camera
                    # angles are present
                    cams_video = [filename.split('_')[-1][3:-4] for filename in video_paths]
                    cams_csv = [filename.split('_')[-1][3:-4] for filename in csv_paths]

                    if set(string_cams).issubset(cams_video):
                        video_paths_all.append(video_paths)
                    else:
                        continue

                    if set(string_cams).issubset(cams_csv):
                        csv_paths_all.append(csv_paths)
                    else:
                        continue

        return video_paths_all, csv_paths_all

    def load_morph_model(self):
        """
        Load the morphing model for the specified model name
        """
        morph_model_path = "skeletonMorphing/models/morph_" + str(self.model_name) + ".pth"
        model_skel_morph = modelSkeletonMorphing.Synthesizer()
        if os.path.isfile(morph_model_path):
            model_skel_morph.load_state_dict(torch.load(morph_model_path))
            model_skel_morph.eval()
        else:
            raise ValueError('Morphing model is missing')

        return model_skel_morph

    def calculate_log_metrics(self, gt_keypoints, pred_keypoints, inference_times):
        """
        Calculate metrics for each sweep and logs them on wandb. Calculates it for the whole body and body segments.
        :param gt_keypoints: [[x0,y0,z0], [x1,y1,z1], [x2,y2,z2], ...]
        :param pred_keypoints: [[x0,y0,z0], [x1,y1,z1], [x2,y2,z2], ...]
        :param inference_times: [inference_time0, inference_time1, inference_time2, ]
        :return:
        """

        # First we calculate the metrics for all extracted keypoints
        mpjpe = metrics.calculate_mpjpe(gt_keypoints, pred_keypoints)
        pmpjpe = metrics.calculate_pmpjpe(gt_keypoints, pred_keypoints)
        pck = metrics.calculate_pck(gt_keypoints, pred_keypoints)
        velocity_error = metrics.mean_velocity_error(gt_keypoints, pred_keypoints)
        acceleration_error = metrics.mean_acceleration_error(gt_keypoints, pred_keypoints)
        cps = metrics.compute_CPS(gt_keypoints, pred_keypoints)
        angular_error = metrics.calculate_mpsae(gt_keypoints, pred_keypoints, self.joint_segments)
        cmc = metrics.calculate_cmc(gt_keypoints, pred_keypoints)

        # Log whole body metrics
        wandb.log({"mpjpe_all": mpjpe, "pmpjpe_all": pmpjpe, "pck": pck, "velocity_error_all": velocity_error,
                   "acceleration_error_all": acceleration_error, "cps": cps, "angular_error_all": angular_error,
                   "cmc_all": cmc})

        # Log metrics for different body segments
        for segment in self.body_segments:
            keypoints = self.body_segments[segment]
            mpjpe = metrics.calculate_mpjpe(gt_keypoints[keypoints], pred_keypoints[keypoints])
            pmpjpe = metrics.calculate_pmpjpe(gt_keypoints[keypoints], pred_keypoints[keypoints])
            pck = metrics.calculate_pck(gt_keypoints[keypoints], pred_keypoints[keypoints])
            velocity_error = metrics.mean_velocity_error(gt_keypoints[keypoints], pred_keypoints)
            acceleration_error = metrics.mean_acceleration_error(gt_keypoints[keypoints], pred_keypoints[keypoints])
            cmc = metrics.calculate_cmc(gt_keypoints[keypoints], pred_keypoints[keypoints])

            wandb.log({"mpjpe_" + segment: mpjpe, "pmpjpe_" + segment: pmpjpe, "pck": pck,
                       "velocity_error_" + segment: velocity_error,
                       "acceleration_error_" + segment: acceleration_error, "cmc_" + segment: cmc})

        # Calculate for each joint separately
        for joint_name in self.joint_segments:
            joint = self.joint_segments[joint_name]
            angular_error = metrics.calculate_mpsae(gt_keypoints, pred_keypoints, joint)

            wandb.log({"angular_error_" + joint_name: angular_error})

        # Log inference time
        inference_time_mean = np.mean(inference_times)
        inference_time_std = np.std(inference_times)
        wandb.log({"inference_time_mean": inference_time_mean, "inference_time_std": inference_time_std})

        # Log number (n) of frames metrics are based on:
        wandb.log({"n": len(pred_keypoints)})

    def plot_n_log(self, gt_keypoints, pred_keypoints):
        # Plot predicted and ground truth keypoints overlay

        # Log the plot to wand
        pass

    def apply_morphing(self, input_pose):
        """
        Apply skeleton morphing to the input pose using the loaded model.
        :param input_pose: Input pose to be morphed.
        :return: morphed_pose: Pose after applying skeleton morphing.
        """
        with torch.no_grad():
            morphed_pose = self.model(input_pose)
        return morphed_pose

    def preprocess_ground_truth(self, csv_path):
        """
        Pre-process the ground truth keypoints by first loading them from the csv file and transform them to be comparable
        with the prediction
        """
        # Read ground truth
        gt_keypoints = []  # Placeholder

        # Any other transformations? Rotation and translation to overlay with image frame? Check HM36 code

        # Morph ground truth
        gt_keypoints = self.model_skel_morph(gt_keypoints)

        return gt_keypoints

    def postprocess_prediction(self, pred_keypoints):
        """
        Post-process the predicted keypoints according to standard methods for real-time applications.
        1) Interpolation of missing values
        2) Smoothing of the data stream
        """
        pred_keypoints_processing = postprocessing.postprocess_points(pred_keypoints,
                                                                      self.interpolation_fun, self.smoothing_fun)

        return pred_keypoints_processing

    def main(self, config=None):
        """
        Run inference on the chosen model with sweep parameters and log results to wandb project.
        """
        with wandb.init(config=config):
            # If called by wandb.agent, as below,
            # this config will be set by Sweep Controller
            config = wandb.config
            # Load video and csv file paths
            video_paths, csv_paths = self.load_data_paths(config)

            # Generate cv2 video API and load respective ground truth keypoints
            caps = []
            gt_keypoints_all = []
            pred_keypoints_all = []
            inference_times_all = []

            # Iterate through all videos specified
            for movement_iter in video_paths:
                # Get all specified cameras of the specified movement iteration and create a cv2 capture stream
                for cam in movement_iter:
                    cap = cv2.VideoCapture(cam)
                    caps.append(cap)
                    # Get ground truth keypoints
                    gt_keypoints = self.preprocess_ground_truth(csv_paths)

                # Open model based on name and run inference
                # Monocular models
                if self.model_type == "mono":
                    if self.model_name == "mediapipe":
                        pred_keypoints, inference_times, frame = mediapipeMono.inference_video(caps, self.sweep_config)
                    if self.model_name == "alphapose":
                        print("AlphaPose not implemented yet")
                        # pred_keypoints, inference_times, frame = alphaPoseMono.inference_video(caps, self)

                # Multioccular models
                elif self.model_type == "multi":
                    # Desynchronize video streams
                    if self.sweep_config['desynchronizer']:
                        caps = self.cam_desynchronizer.desynchronize(caps)
                    # Load camera parameter matrix and add noise if specified so
                    p_matrix = camCali.get_projection_matrix(self.sweep_config['cameras'],
                                                             self.sweep_config['decalibration'])

                    if self.model_name == "canonpose":
                        print("CanonPose not implemented yet")
                        # pred_keypoints, inference_times = canonPoseMulti.inference_video(caps)

                # Do some post-processing on the predicted keypoints
                pred_keypoints = self.postprocess_prediction(pred_keypoints)

                # Collect pred_keypoints for each movement iteration
                gt_keypoints_all.append[gt_keypoints]
                pred_keypoints_all.append[pred_keypoints]
                inference_times_all.append[inference_times]

            # Calculate the metrics, generate plots and log them to wandb
            self.calculate_log_metrics(gt_keypoints, pred_keypoints, inference_times)
            self.plot_n_log(gt_keypoints, pred_keypoints)
            log_frame_example(frame)

    def initiate_wandb_sweep(self):
        """
        Get settings from the parser and initiate the sweep with all parameters
        :return:
        """

        # Sanity check of inputs
        if self.model_type not in ['multi', 'mono']:
            raise ValueError('Choose a valid model type (multi or mono)')

        if self.model_type == 'mono':
            if self.model_name not in ['mediapipe', 'alphapose', 'unknown']:
                raise ValueError('Choose a valid monooccular model, or change the model type to multi')
        elif self.model_type == 'multi':
            if self.model_name not in ['cdrnet', 'canonpose']:
                raise ValueError('Choose a valid multioccular model, or change the model type to mono')

        if self.dataset_path is None:
            raise ValueError('Dataset path is missing')

        # Set sweep config to grid search, which iterates over every possible combination
        self.sweep_config = {
            'name': 'sweep_' + self.model_type + '_' + self.model_name,
            'dataset': self.participants,
            'method': 'grid',
            'parameters': {
                'movement': {
                    'values': ['upper', 'lower', 'sitting', 'complex']
                }
            }
        }

        # If there are multiple cameras updates sweep parameters:
        if self.model_type == 'mono':
            # Sweep parameters
            parameters_dict = {
                'augmentation': {
                    'values': ['defocus', 'underexposure', 'motion_blur', 'occlusion', 'background']
                },
                'cameras': {
                    'values': [0, 5, 4, 1, 3, 2]
                }
            }
        elif self.model_type == 'multi':
            # Sweep parameters
            parameters_dict = {
                'augmentation': {
                    'values': ['defocus', 'underexposure', 'motion_blur', 'occlusion', 'background',
                               'desynchronize', 'decalibration']
                },
                'cameras': {
                    'values': [[4, 0], [3, 2], [5, 1], [4, 2], [0, 4, 3], [0, 2, 3], [5, 4, 1], [0, 4, 3, 2],
                               [0, 5, 4, 3, 2],
                               [0, 5, 4, 1, 3, 2]]
                }
            }

        self.sweep_config['parameters'].update(parameters_dict)

        # Initialize the sweep run
        sweep_id = wandb.sweep(sweep=self.sweep_config,
                               project='HPE_framework')

        # Start sweep
        wandb.agent(sweep_id, function=self.main)

        # Training complete
        print("Testing complete")

# Run the framework
Framework(model_name="mediapipe", model_type="mono", directory="/home/emanu/Desktop/SegmentedData").initiate_wandb_sweep()

import os
import wandb
import numpy as np
import cv2
from utils import cameraCalibrationAugmentation as caliAug
from utils import frameAugmentation as frameAug
from utils import metrics


class Framework:
    def __init__(self, model_name, model_type, dataset_path):
        # Sanity check of inputs
        if model_type not in ['multi', 'mono']:
            raise ValueError('Choose a valid model type (multi or mono)')

        if model_type == 'mono':
            if model_name not in ['mediapipe', 'alphapose', 'unknown']:
                raise ValueError('Choose a valid monooccular model, or change the model type to multi')
        elif model_type == 'multi':
                raise ValueError('Choose a valid multioccular model, or change the model type to mono')

        if dataset_path is None:
            raise ValueError('Dataset path is missing')

        self.dataset_path = dataset_path

        participants = ['par5', 'par6', 'par7', 'par8', 'par9', 'par10', 'par11', 'par12', 'par14', 'par15',
                        'par16', 'par17', 'par18', 'par19', 'par20', 'par21', 'par22', 'par23', 'par24', 'par25',
                        'par26']

        # Set sweep config to grid search, which iterates over every possible combination
        self.sweep_config = {
            'name': 'sweep_'+model_type+'_'+model_name,
            'dataset': participants,
            'method': 'grid'}

        # Set parameters and values for the sweep
        parameters_dict = {
            'parameters':{
                'defocus': {
                    'values': [True, False]
                    },
                'underexposure': {
                    'values': [True, False]
                    },
                'overexposure': {
                      'values': [True, False]
                    },
                'motion_blur': {
                    'values': [True, False]
                },
                'occlusion': {
                    'values': [True, False]
                },
                'background': {
                    'values': ['none', 'home', 'hospital', 'outdoor', 'people']
                },
                'movement': {
                    'values': ['upper', 'lower', 'sitting', 'complex']
                },
                'cameras': {
                    'values': [0, 5, 4, 1, 3, 2]
                }
            }
        }

        # If there are multiple cameras updates sweep parameters:
        if model_type == 'multi':
            # Sweep parameters
            multioccular_parameters = {
                    'desynchronize': {
                        'values': [True, False]
                    },
                    'decalibration': {
                        'values': [True, False]
                    },
                    'cameras': {
                      'values': [[4, 0], [3, 2], [5, 1], [4, 2], [0, 4, 3], [0, 2, 3], [5, 4, 1], [0, 4, 3, 2], [0, 5, 4, 3, 2],
                                 [0, 5, 4, 1, 3, 2]]
                    }
            }
            parameters_dict['parameters'].update(multioccular_parameters)

        # Combine the sweep configuration and sweep parameters
        self.sweep_config.update(parameters_dict)

        # Initialize the sweep run
        self.sweep_id = wandb.sweep(sweep=self.sweep_config,
                         project='HPE_framework',
                         description='Clinical Evaluation of different real-time 3D HPE models')

        # Start sweep. Might should be at the end... not sure yet
        wandb.agent(self.sweep_id, function=Framework.main())

    def augment_frames(self, frames):
        frames_aug = []
        for frame in frames:
            if self.sweep_config['background'] != 'none':
                frame = self.augmenter.BackgroundChanger(frame, self.sweep_config['background'])
            if self.sweep_config['motion_blur']:
                frame = frameAug.motion_blur(frame)
            if self.sweep_config['occlusion']:
                frame = frameAug.occlusion(frame)
            if self.sweep_config['defocus']:
                frame = frameAug.defocus(frame)
            if self.sweep_config['underexposure']:
                frame = frameAug.underexposure(frame)

            frames_aug.append(frame)

        return frames_aug

    def load_data_paths(self):
        participants = self.config.sweep_config['dataset']
        mov_cats = self.sweep_config['movement']
        cams = self.sweep_config['cameras']
        string_cams = list(map(str, cams))
        iterations = np.arange(1, 11)

        # Translate categories into the respective dataset numbers
        if mov_cats == 'upper':
            movement_nr = [1, 2, 3, 4]
        elif mov_cats == 'lower':
            movement_nr = [5, 6, 7, 8]
        elif mov_cats == 'complex':
            movement_nr = [9, 10, 11, 12, 13]
        elif mov_cats == 'sitting':
            movement_nr = [14, 15, 16, 17]

        # Loop through all the video and csv files and extract the for the sweep relevant ones
        video_paths_all = []
        csv_paths_all = []

        for participant in participants: # Loop through participants
            folder_path = os.path.join(self.dataset_path, participant)
            if os.path.isdir(folder_path): # Check if it's a directory
                    for mov_cat in mov_cats:  # Loop for each chosen movement type
                        for iteration in iterations: # Loop for each iteration
                            video_paths = [] # Reset singel iteration arrays
                            csv_paths = []
                            for cam in cams:  # Loop for each camera
                                video_file_name = f"{participant}_Mov{mov_cat}_Iter{iteration}_Cam{cam}.avi"
                                csv_file_name = f"{participant}_Mov{mov_cat}_Iter{iteration}_Cam{cam}.csv"
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


    # Calculate relevant metrics and log them to wandb.
    def calculate_log_metrics(self, gt_keypoints, pred_keypoints):
        # First we calculate the metrics for all extracted keypoints
        mpjpe = metrics.calculate_mpjpe(gt_keypoints, pred_keypoints)
        pmpjpe = metrics.calculate_pmpjpe(gt_keypoints, pred_keypoints)
        velocity_error = metrics.mean_velocity_error(gt_keypoints, pred_keypoints)
        acceleration_error = metrics.mean_acceleration_error(gt_keypoints, pred_keypoints)
        cps = metrics.compute_CPS(gt_keypoints, pred_keypoints)
        angular_error = metrics.calculate_mpsae(gt_keypoints, pred_keypoints)
        rom = metrics.calculate_rom(gt_keypoints, pred_keypoints)
        cmc = metrics.calculate_cmc(gt_keypoints, pred_keypoints)

        # Log whole body metrics
        self.wandb.log({"mpjpe_all": mpjpe, "pmpjpe_all": pmpjpe, "velocity_error_all": velocity_error,
                        "acceleration_error_all": acceleration_error, "cps": cps, "angular_error_all": angular_error,
                        "rom_all": rom, "cmc_all": cmc})

        # Log metrics for different body
        segment_names = {
            "left_lower_arm": [],
            "left_upper_arm": [],
            "right_lower_arm": [],
            "right_upper_arm": [],
            "torso": [],
            "left_upper_leg": [],
            "left_lower_leg": [],
            "right_upper_leg": [],
            "right_lower_leg": []}

        for segment in self.segments:
            points = segment
            mpjpe = metrics.calculate_mpjpe(gt_keypoints[points], pred_keypoints[points])
            pmpjpe = metrics.calculate_pmpjpe(gt_keypoints[points], pred_keypoints[points])
            velocity_error = metrics.mean_velocity_error(gt_keypoints[points], pred_keypoints)
            acceleration_error = metrics.mean_acceleration_error(gt_keypoints[points], pred_keypoints[points])
            cps = metrics.compute_CPS(gt_keypoints[points], pred_keypoints[points])
            angular_error = metrics.calculate_mpsae(gt_keypoints[points], pred_keypoints[points])
            rom = metrics.calculate_rom(gt_keypoints[points], pred_keypoints[points])
            cmc = metrics.calculate_cmc(gt_keypoints[points], pred_keypoints[points])

            self.wandb.log({"mpjpe_"+segment: mpjpe, "pmpjpe_"+segment: pmpjpe,
                            "velocity_error_"+segment: velocity_error,
                            "acceleration_error_"+segment: acceleration_error,
                            "angular_error_"+segment: angular_error, "rom_"+segment: rom, "cmc_"+segment: cmc})

        # Log inference time
        inference_time_mean = np.mean(inferences_time)
        inference_time_std = np.std(inferences_time)
        self.wandb.log({"inference_time_mean": inference_time_mean, "inference_time_std": inference_time_std})





        def preprocess_ground_truth(self):

        ## Postprocess keypoints
        # Post processing the predicted keypoints using standard methods.
        def postprocess_prediction(self, keypoints):




    # Run inference on the chosen model
    def main(self):
        # Initialize video augmenters
        if self.sweep_config['background'] != 'none':
            self.augmenter = frameAug.BackgroundChanger()

        if self.sweep_config['desynchronize']:
            self.cam_desynchronizer = frameAug.CameraDesynchronizer()

        if self.sweep_config['decalibration']:
            caliAug.calibration_noise([self.R, self.t])


        run = self.wandb.init()
    # Open model based on name

    # Load video files

    #

    # Run inference
        caps = []

        cap = cv2.VideoCapture(input_video_path)


        # Desynchronize video streams
        if self.sweep_config['desynchronizer']:
            caps = self.cam_desynchronizer.desynchronize(caps)

        caps.append(cap)

        return caps, keypoints
    # Hand it

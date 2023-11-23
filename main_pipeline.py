import wandb
import numpy as np
import augmentation.calibrationAugmentation as caliaug
import augmentation.frameAugmentation as frameaug
import skeletonMorphing.applySkeletonMorphing as skeletonmorph


class Framework:
    def __init__(self, model_name, model_type):
        # Sanity check of inputs
        if model_type not in ['multi', 'mono']:
            raise ValueError('Choose a valid model type (multi or mono)')

        if model_type == 'mono':
            if model_name not in ['mediapipe', 'alphapose', 'unknown']:
                raise ValueError('Choose a valid monooccular model, or change the model type to multi')
        elif model_type == 'multi':
                raise ValueError('Choose a valid multioccular model, or change the model type to mono')

        # Set sweep config to grid search, which iterates over every possible combination
        self.sweep_config = {
            'name': 'sweep_'+model_type+'_'+model_name,
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
                    'values': ['home', 'hospital', 'outdoor', 'people']
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






    def load_data(self):
        mov_cat = self.sweep_config['movement']
        cam = self.sweep_config['cameras']

        if mov_cat == 'upper':
            movement_nr = [1, 2, 3, 4];
        elif mov_cat == 'lower':
            movement_nr = [5, 6, 7, 8];
        elif mov_cat == 'complex':
            movement_nr = [9, 10, 11, 12, 13];
        elif mov_cat == 'sitting':
            movement_nr = [14, 15, 16, 17];

    def augment_frames(self,):


    def preprocess_ground_truth(self,):


    ## Postprocess keypoints
    # Post processing the predicted keypoints using standard methods.
    def postprocess_prediction():




    ## Calculate and log metrics
    # Calculate relevant metrics and log them to wandb.
    def calculate_log_metrics():
        # Log whole body metrics
        wandb.log({"mpjpe_all": mpjpe, "pmpjpe_all": pmpjpe, "velocity_error_all": velocity_error,
                   "angular_error_all": angular_error, "rom_all": rom, "cmc_all": cmc})

        # Log metrics for different body
        segment_names = {
            "left_lower_arm",
            "left_upper_arm",
            "right_lower_arm",
            "right_upper_arm",
            "torso",
            "left_upper_leg",
            "left_lower_leg",
            "right_upper_leg",
            "right_lower_leg"]

        for segment in segment_names:
            mpjpe_value = calculate_mpjpe_for_segment(segment)
            pmpjpe_value = calculate_pmpjpe_for_segment(segment)
            velocity_error = calculate_pmpjpe_for_segment(segment)
            angular_error = calculate_pmpjpe_for_segment(segment)
            rom = calculate_pmpjpe_for_segment(segment)
            cmc = calculate_pmpjpe_for_segment(segment)

            wandb.log({"mpjpe_"+segment: mpjpe_value, "pmpjpe_"+segment: pmpjpe_value,
                                        "velocity_error_"+segment: velocity_error, "angular_error_"+segment: angular_error,
                                        "rom_"+segment: rom, "cmc_"+segment: cmc})

        # Log overall correct pose score
        wandb.log({"correct_pose_score": cps})

        # Log inference time
        inference_time_mean = np.mean(inferences_time)
        inference_time_std = np.std(inferences_time)
        wandb.log({"inference_time_mean": inference_time_mean, "inference_time_std": inference_time_std})

    # Run inference on the chosen model
    def main(self):
    run = self.wandb.init()
    # Open model based on name

    # Load video files

    #

    # Run inference

    # Hand it

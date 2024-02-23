import os
import wandb
import numpy as np
import torch
from tqdm import tqdm
from skeletonMorphing import modelSkeletonMorphing
from utils import cameraCalibration as camCali
from utils import frameAugmentation as frameAug
from utils.plotKeypoints import plot_3d_keypoints_gt_pred
from utils import metrics, postprocessing, readDataEval
from models import mediapipeMono

def log_frame_example(frames):
    """
    Log the last frame of one camera angle to visualize applied frame augmentations
    :param frames: augmented frames, e.g. [cam0, cam1, cam2, ...]
    """
    image = wandb.Image(frames, caption=f"Frame example of current evaluation")
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
        # self.participants = ['par6', 'par7', 'par8', 'par9', 'par10', 'par11', 'par12', 'par14', 'par15',
        #                     'par16', 'par17', 'par18', 'par19', 'par20', 'par21', 'par22', 'par23', 'par24', 'par25',
        #                     'par26']
        self.participants = ['par5']

        # Defines movement number in dataset related to different movement categories
        self.movement_category = {
            "upper": [1, 2, 3, 4],
            "lower": [5, 6, 7, 8],
            "complex": [9, 10, 11, 12, 13],
            "sitting": [14, 15, 16, 17]
        }

        # Defines body segments
        self.body_segments = {
            "left_lower_arm": [3, 5],
            "left_upper_arm": [1, 3],
            "right_lower_arm": [2, 4],
            "right_upper_arm": [0, 2],
            # "torso": [0, 1, 7, 6],
            "left_upper_leg": [7, 9],
            "left_lower_leg": [9, 11],
            "right_upper_leg": [6, 8],
            "right_lower_leg": [8, 12],
            "left_foot": [13, 15],
            "right_foot": [12, 14]}

        # Defines pairs of bones for each body segment of each side of the body
        self.bone_pairs = {
            "lower_arm": [0, 2],
            "upper_arm": [1, 3],
            "upper_leg": [4, 6],
            "lower_leg": [5, 7],
            "foot": [8, 9]
        }

        # Defines segments around each joint to calculate angles (distal to proximal).
        self.joint_segments = {
            "right_elbow": [0, 2, 4],
            "left_elbow": [1, 3, 5],
            "right_shoulder_1": [2, 0, 1],
            "right_shoulder_2": [2, 0, 6],
            "left_shoulder_1": [3, 1, 0],
            "left_shoulder_2": [3, 1, 7],
            "right_hips_1": [8, 6, 7],
            "right_hips_2": [8, 6, 0],
            "left_hips_1": [9, 7, 8],
            "left_hips_2": [9, 7, 1],
            "right_knee": [6, 8, 10],
            "left_knee": [7, 9, 11],
            "right_ankle": [12, 10, 14],
            "left_ankle": [13, 11, 15]
        }

        self.interpolation_fun = "akima"
        self.smoothing_fun = "median"

    def load_morph_model(self):
        """
        Load the morphing model for the specified model name
        :return model_skel_morph: Morphing model for the specified model name
        """
        morph_model_path = "skeletonMorphing/models/morph_" + str(self.model_name) + ".pth"
        model_skel_morph = modelSkeletonMorphing.Synthesizer()
        if os.path.isfile(morph_model_path):
            model_skel_morph.load_state_dict(torch.load(morph_model_path))
            model_skel_morph.eval()
        else:
            raise ValueError('Morphing model is missing')

        return model_skel_morph

    def apply_morphing(self, input_pose):
        """
        Apply skeleton morphing to the input pose using the loaded model.
        :param input_pose: Input pose to be morphed as numpy array.
        :return: morphed_pose: Pose after applying skeleton morphing.
        """
        # Convert input pose to tensor and apply morphing
        input_pose = torch.from_numpy(input_pose)
        inp_poses = input_pose.view(-1, input_pose.size(1) * input_pose.size(2)).float()
        with torch.no_grad():
            morphed_pose = self.model_skel_morph(inp_poses)
        return morphed_pose.view(-1, input_pose.size(1), input_pose.size(2)).cpu().detach().numpy()

    def calculate_log_metrics(self, gt_keypoints, pred_keypoints, inference_times):
        """
        Calculate metrics for each sweep and logs them on wandb. Calculates it for the whole body and body segments.
        :param gt_keypoints: [[x0,y0,z0], [x1,y1,z1], [x2,y2,z2], ...]
        :param pred_keypoints: [[x0,y0,z0], [x1,y1,z1], [x2,y2,z2], ...]
        :param inference_times: [inference_time0, inference_time1, inference_time2, ]
        """

        # First we calculate the metrics for all extracted keypoints
        # The metric for distance is millimeters
        mpjpe_m, mpjpe_s = metrics.calculate_mpjpe(gt_keypoints, pred_keypoints)
        pmpjpe_m, pmpjpe_s = metrics.calculate_pmpjpe(gt_keypoints, pred_keypoints)
        pck = metrics.calculate_pck(gt_keypoints, pred_keypoints)
        velocity_m, velocity_s = metrics.mean_velocity_error(gt_keypoints, pred_keypoints)
        cps, cps_auc = metrics.compute_CPS(gt_keypoints, pred_keypoints)
        angular_m, angular_s, angular_m_s, angular_s_s = metrics.calculate_mpsae(gt_keypoints, pred_keypoints, self.joint_segments)
        cmc, pvalue = metrics.calculate_cmc(gt_keypoints, pred_keypoints)

        # Log whole body metrics
        wandb.log({"mpjpe_all_mean": mpjpe_m, "mpjpe_all_std": mpjpe_s,
                   "pmpjpe_all_mean": pmpjpe_m, "pmpjpe_all_std": pmpjpe_s,
                   "velocity_error_all_mean": velocity_m, "velocity_error_all_std": velocity_s,
                   "pck_all": pck,
                   "cps_all": cps, "cps_auc_all": cps_auc,
                   "angular_all_mean": angular_m, "angular_all_std": angular_s,
                   "angular_seg_mean": angular_m_s, "angular_seg_std": angular_s_s,
                   "cmc_all": cmc, "cmc_pvalue_all": pvalue})

        # Bone symmetry MISSING

        # Log metrics for different body segments
        for segment in self.body_segments:
            keypoints = self.body_segments[segment]
            mpjpe_m, mpjpe_s = metrics.calculate_mpjpe(gt_keypoints[keypoints], pred_keypoints[keypoints])
            pmpjpe_m, pmpjpe_s = metrics.calculate_pmpjpe(gt_keypoints[keypoints], pred_keypoints[keypoints])
            # pck = metrics.calculate_pck(gt_keypoints[keypoints], pred_keypoints[keypoints])
            velocity_m, velocity_s = metrics.mean_velocity_error(gt_keypoints[keypoints], pred_keypoints[keypoints])
            cmc, pvalue = metrics.calculate_cmc(gt_keypoints[keypoints], pred_keypoints[keypoints])

            # Bone symmetry MISSING

            wandb.log({"mpjpe_mean_"+segment: mpjpe_m, "mpjpe_std_"+segment: mpjpe_s,
                       "pmpjpe_mean_"+segment: pmpjpe_m, "pmpjpe_std_"+segment: pmpjpe_s,
                       "velocity_mean_"+segment: velocity_m, "velocity_std_"+segment: velocity_s,
                       # "pck_"+segment: pck,
                       "cmc_"+segment: cmc, "cmc_pvalue_"+segment: pvalue})

        # Log inference time
        inference_time_mean = np.mean(inference_times)
        inference_time_std = np.std(inference_times)
        wandb.log({"inference_time_mean": inference_time_mean, "inference_time_std": inference_time_std})

        # Log number (n) of frames metrics are based on:
        wandb.log({"sample_number": len(pred_keypoints)})

    def main(self, config=None):
        """
        Run inference on the chosen model with sweep parameters and log results to wandb project.
        """
        with wandb.init(config=config) as run:
            # If called by wandb.agent, as below,
            # this config will be set by Sweep Controller
            config = wandb.config

            # Overwrite the random name of the run with the sweep name
            run.name = config.model_name + "-" + config.movement + "-" + config.augmentation + "-" + str(config.cameras)

            # Generate cv2 video API and load respective ground truth keypoints
            gt_keypoints_all = []
            pred_keypoints_all = []
            inference_times_all = []

            for par in tqdm(self.participants, ascii=True, desc="Participants"):
                for mov in tqdm(self.movement_category[config['movement']], ascii=True, desc="Movements"):
                    #try:
                    gt_keypoints, caps = readDataEval.load_data(self.dataset_path, par, mov, config['cameras'],
                                                                self.model_name)
                    #except:
                    #    print(f"Participant {par} and movement {mov} not found")
                    #    continue

                    # Open model based on name and run inference
                    # Monocular models
                    if self.model_type == "mono":
                        # Just load the first (and only) camera
                        gt_keypoints = gt_keypoints[0][1]
                        caps = caps[0][1]

                        if self.model_name == "mediapipe":
                            pred_keypoints, inference_times, frame = mediapipeMono.inference_video(caps, config)
                            selected_columns = [12, 11, 14, 13, 16, 15, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31]
                            pred_keypoints = pred_keypoints[:, selected_columns, 1:]

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

                    # Postprocess predicted keypoints
                    pred_keypoints_processing = postprocessing.postprocess_points(pred_keypoints,
                                                                                  self.interpolation_fun,
                                                                                  self.smoothing_fun)

                    # Morph ground truth to format of predicted keypoints
                    pred_keypoints_morphed = self.apply_morphing(pred_keypoints_processing)

                    # Collect pred_keypoints for each movement iteration
                    gt_keypoints_all.append(gt_keypoints)
                    pred_keypoints_all.append(pred_keypoints_morphed)
                    inference_times_all.append(inference_times)

            # Calculate the metrics, generate plots and log them to wandb
            #with open('results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            #    pickle.dump([gt_keypoints_all, pred_keypoints_all, inference_times_all], f)
            inference_times_all = np.concatenate(inference_times_all)
            gt_keypoints_all = np.concatenate(gt_keypoints_all, axis=0)
            pred_keypoints_all = np.concatenate(pred_keypoints_all, axis=0)
            self.calculate_log_metrics(gt_keypoints_all, pred_keypoints_all, inference_times_all)
            plot_3d_keypoints_gt_pred(gt_keypoints[-1], pred_keypoints_morphed[-1], self.model_name)
            #log_frame_example(frame)

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
            'method': 'grid',
            'program': 'main_pipeline_debug.py',
            'parameters': {
                'model_type': {
                    'value': [self.model_type]
                },
                'model_name': {
                    'value': [self.model_name]
                },
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
                               project='HPE_framework_2')

        # Start sweep
        wandb.agent(sweep_id, function=self.main)

        # Training complete
        print("Testing complete")


# Run the framework
Framework(model_name="mediapipe", model_type="mono",
          directory="/home/emanu/Desktop/SegmentedData").initiate_wandb_sweep()

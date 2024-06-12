import os
import wandb
import numpy as np
import torch
import cv2
from tqdm import tqdm
from skeletonMorphing import modelSkeletonMorphing
from utils import cameraCalibration as camCali
from utils import frameAugmentation as frameAug
from utils.plotKeypoints import plot_3d_keypoints_gt_pred
from utils import metrics, postprocessing, readDataEval
from models import mediapipeMono
import pickle


def log_frame_example(frame):
    """
    Log the last frame of one camera angle to visualize applied frame augmentations. Blur the faces for privacy.
    :param frame: augmented frames, e.g. [cam0, cam1, cam2, ...]
    """

    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    face_data = face_detect.detectMultiScale(frame, 1.2, 3)

    for (x, y, w, h) in face_data:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = frame[y:y + h, x:x + w]
        # applying a gaussian blur over this new rectangle area
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        # impose this blurred image on original image to get final image
        frame[y:y + roi.shape[0], x:x + roi.shape[1]] = roi

    image = wandb.Image(frame)
    wandb.log({"Example_Frame": image})


class Framework:
    def __init__(self, model_name, model_type, sample_rate, directory):
        # Access the parsed arguments
        self.model_name = model_name
        self.model_type = model_type
        self.sample_rate = sample_rate
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
        # self.body_segments = {
        #    "left_lower_arm": [3, 5],
        #    "left_upper_arm": [1, 3],
        #    "right_lower_arm": [2, 4],
        #    "right_upper_arm": [0, 2],
        #    "left_upper_leg": [7, 9],
        #    "left_lower_leg": [9, 11],
        #    "right_upper_leg": [6, 8],
        #    "right_lower_leg": [8, 12],
        #    "left_foot": [13, 15],
        #    "right_foot": [12, 14]}

        self.body_segments = {
            "left_lower_arm": ['elbow_left', 'wrist_left'],
            "left_upper_arm": ['shoulder_left', 'elbow_left'],
            "right_lower_arm": ['elbow_right', 'wrist_right'],
            "right_upper_arm": ['shoulder_right', 'elbow_right'],
            "left_upper_leg": ['hip_left', 'knee_left'],
            "left_lower_leg": ['knee_left', 'ankle_left'],
            "right_upper_leg": ['hip_right', 'knee_right'],
            "right_lower_leg": ['knee_right', 'ankle_right'],
            "left_foot": ['ankle_left', 'toe_left'],
            "right_foot": ['ankle_right', 'toe_right']
        }

        # Joint names mapping for MoCap ground truth
        self.joint_names_gt = {
            0: 'right_shoulder', 1: 'left_shoulder', 2: 'right_elbow', 3: 'left_elbow',
            4: 'right_wrist', 5: 'left_wrist', 6: 'right_hip', 7: 'left_hip',
            8: 'right_knee', 9: 'left_knee', 10: 'right_ankle', 11: 'left_ankle',
            12: 'right_heel', 13: 'left_heel', 14: 'right_foot_index', 15: 'left_foot_index'
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

        ###########################################
        # MAYBE CALCULATE PROCRUSTES BEFORE ONCE ##
        ###########################################

        # Prepare input
        keypoint_names = list(gt_keypoints[0].keys())
        keypoint_names_pred = list(pred_keypoints[0].keys())
        assert keypoint_names == keypoint_names_pred, "Both the labels of the prediction and ground truth must match."
        gt_keypoints = [[sample[key] for key in keypoint_names] for sample in gt_keypoints]
        pred_keypoints = [[sample[key] for key in keypoint_names] for sample in pred_keypoints]
        gt_keypoints = [item for sublist in gt_keypoints for item in sublist]
        pred_keypoints = [item for sublist in pred_keypoints for item in sublist]

        # Spatial metrics
        pmpjpe_m, pmpjpe_s, proc_error = metrics.calculate_pmpjpe(gt_keypoints, pred_keypoints)
        angular_m, angular_s, r2 = metrics.calculate_angle_error(gt_keypoints, pred_keypoints)
        cps, cps_auc = metrics.compute_CPS(gt_keypoints, pred_keypoints)
        # Spatio-temporal metrics
        velocity_m, velocity_s = metrics.mean_velocity_error(gt_keypoints, pred_keypoints, self.sample_rate)
        pcc, pvalue = metrics.calculate_correlation(gt_keypoints, pred_keypoints)
        # Cosine similarity?

        # Log whole body metrics
        wandb.log({"pmpjpe_all_mean": pmpjpe_m, "pmpjpe_all_std": pmpjpe_s,
                   "angular_all_mean": angular_m, "angular_all_std": angular_s,
                   "cps_all": cps, "cps_auc_all": cps_auc,
                   "velocity_error_all_mean": velocity_m, "velocity_error_all_std": velocity_s,
                   "pcc_all": pcc, "pvalue_all": pvalue,
                   "procrustes_n_all": len(pred_keypoints) - proc_error,
                   })

        # Log metrics for different body segments
        for segment in self.body_segments:
            keypoints = self.body_segments[segment]
            pmpjpe_m, pmpjpe_s, proc_error = metrics.calculate_pmpjpe(gt_keypoints[keypoints], pred_keypoints[keypoints])
            velocity_m, velocity_s = metrics.mean_velocity_error(gt_keypoints[keypoints], pred_keypoints[keypoints])
            pcc, pvalue = metrics.calculate_correlation(gt_keypoints[keypoints], pred_keypoints[keypoints])

            wandb.log({"pmpjpe_mean_"+segment: pmpjpe_m, "pmpjpe_std_"+segment: pmpjpe_s,
                       "velocity_mean_"+segment: velocity_m, "velocity_std_"+segment: velocity_s,
                       "pcc_"+segment: pcc, "pvalue_"+segment: pvalue,
                       "procrustes_n_"+segment: len(pred_keypoints) - proc_error})

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

            for par in tqdm(self.participants, ascii=True, desc="Participant:"):
                for mov in tqdm(self.movement_category[config['movement']], ascii=True, desc="Movement:"):
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
                            selected_columns = [12, 11, 14, 13, 16, 15, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31] # Select only relevant keypoints and put them in the right order
                            pred_keypoints = pred_keypoints[:, selected_columns, 1:]

                            # Joint names mapping for MediaPipe
                            self.joint_names = {
                                0: 'right_shoulder', 1: 'left_shoulder', 2: 'right_elbow', 3: 'left_elbow',
                                4: 'right_wrist', 5: 'left_wrist', 6: 'right_hip', 7: 'left_hip',
                                8: 'right_knee', 9: 'left_knee', 10: 'right_ankle', 11: 'left_ankle',
                                12: 'right_heel', 13: 'left_heel', 14: 'right_foot_index', 15: 'left_foot_index'
                            }

                        if self.model_name == "motionbert":
                            print("MotionBERT not implemented yet")
                            # pred_keypoints, inference_times, frame = alphaPoseMono.inference_video(caps, self)

                    # Multioccular models
                    elif self.model_type == "multi":
                        # Desynchronize video streams
                        if self.sweep_config['desynchronizer']:
                            caps = self.cam_desynchronizer.desynchronize(caps)
                            # Load camera parameter matrix and add noise if specified so
                            p_matrix = camCali.get_projection_matrix(self.sweep_config['cameras'],
                                                                     self.sweep_config['decalibration'])
                        if self.model_name == "openpose":
                            print("OpenPose not implemented yet")
                            # pred_keypoints, inference_times = canonPoseMulti.inference_video(caps)

                        if self.model_name == "LWCDR":
                            print("LWCDR not implemented yet")
                            # pred_keypoints, inference_times = canonPoseMulti.inference_video(caps)

                    # Postprocess predicted keypoints
                    pred_keypoints = postprocessing.postprocess_points(pred_keypoints,
                                                                                  self.interpolation_fun,
                                                                                  self.smoothing_fun)

                    # Morph ground truth to format of predicted keypoints (can't handle gaps)
                    pred_keypoints = self.apply_morphing(pred_keypoints)

                    # Add joint names
                    pred_keypoints = {self.joint_names[i]: pred_keypoints[:, i, :] for i in
                                      range(pred_keypoints.shape[1])}

                    gt_keypoints = {self.joint_names_gt[i]: gt_keypoints[:, i, :] for i in
                                      range(gt_keypoints.shape[1])}

                    # Collect pred_keypoints for each movement iteration
                    gt_keypoints_all.append(gt_keypoints)
                    pred_keypoints_all.append(pred_keypoints)
                    inference_times_all.append(inference_times)

                    # Collect last good frame for visualization
                    if frame is not None or frame.size != 0:
                        frame_example = frame

            # Save data for debugging
            data_to_save= {'gt_keypoints': gt_keypoints_all,
                           'pred_keypoints': pred_keypoints_all,
                           'inference_time': inference_times_all,
                           'frame': frame_example
                           }

            with open(run.name + '.pkl', 'wb') as file:
                 pickle.dump(data_to_save, file)


            # Calculate the metrics, generate plots and log them to wandb
            with open('results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([gt_keypoints_all, pred_keypoints_all, inference_times_all], f)
            print("Calculating metrics and logging to wandb")
            inference_times_all = np.concatenate(inference_times_all)
            gt_keypoints_all = np.concatenate(gt_keypoints_all, axis=0)
            pred_keypoints_all = np.concatenate(pred_keypoints_all, axis=0)
            self.calculate_log_metrics(gt_keypoints_all, pred_keypoints_all, inference_times_all)
            # plot metrics

            plot_3d_keypoints_gt_pred(gt_keypoints[-1], pred_keypoints_all[-1], self.model_name)
            log_frame_example(frame_example)
            # Log r2
            print("Calculating metrics and logging to wandb")

            # Save metrics for each movement for later calculations

    def initiate_wandb_sweep(self):
        """
        Get settings from the parser and initiate the sweep with all parameters
        :return:
        """

        # Sanity check of inputs
        if self.model_type not in ['multi', 'mono']:
            raise ValueError('Choose a valid model type (multi or mono)')

        if self.model_type == 'mono':
            if self.model_name not in ['mediapipe', 'motionbert']:
                raise ValueError('Choose a valid monooccular model, or change the model type to multi')
        elif self.model_type == 'multi':
            if self.model_name not in ['openpose', 'LWCDR']:
                raise ValueError('Choose a valid multioccular model, or change the model type to mono')

            if self.sample_rate is None:
                raise ValueError('Sample rate missing (Hz)')

        if self.dataset_path is None:
            raise ValueError('Dataset path is missing')

        # Set sweep config to grid search, which iterates over every possible combination
        self.sweep_config = {
            'method': 'grid',
            'program': 'main_pipeline_debug.py',
            'parameters': {
                'model_type': {
                    'value': self.model_type
                },
                'model_name': {
                    'value': self.model_name
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
                    'values': [[4, 0], [5, 1], [4, 3], [4, 2], [0, 4, 3], [5, 4, 1], [0, 4, 3, 2]]
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
Framework(model_name="mediapipe", model_type="mono", sample_rate=25,
          directory="/media/emanu/LaCie/MoCap/segmented").initiate_wandb_sweep()

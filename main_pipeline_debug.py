import os
import wandb
import numpy as np
import torch
import cv2
from tqdm import tqdm
from skeletonMorphing import modelSkeletonMorphing
from utils import cameraCalibration as camCali
from utils import frameAugmentation as frameAug
from utils.plotKeypoints import plot_3d_keypoints_validation
from utils import metrics, postprocessing, readDataEval
from models import mediapipeMono
import pickle
from scipy.spatial import procrustes


def log_frame_example(frame):
    """
    Log the last frame of one camera angle to visualize applied frame augmentations. Blur the faces for privacy.
    :param frame: augmented frames, e.g. [cam0, cam1, cam2, ...]
    """
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    face_data = face_detect.detectMultiScale(frame, 1.2, 3)

    for (x, y, w, h) in face_data:
        roi = frame[y:y + h, x:x + w]
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        frame[y:y + roi.shape[0], x:x + roi.shape[1]] = roi

    image = wandb.Image(frame)
    wandb.log({"Example_Frame": image})


def align_procrustes(target, prediction):
    """
    Procrustes MPJPE: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    Based on the implementation from https://github.com/miraymen/3dpw-eval/blob/master/evaluate.py
    :param target: Ground truth 3D joint positions, shape [sample, joint, 3]
    :param prediction: Predicted 3D joint positions, shape [sample, joint, 3]
    :return gt_all: Ground truth 3D joint positions after alignment, shape [sample, joint, 3]
    :return pred_all: Predicted 3D joint positions after alignment, shape [sample, joint, 3]
    :return error_count: Error count of failed alignments
    """
    assert target.shape[0] == prediction.shape[0], 'Input must have the same number of frames'

    gt_all = []
    pred_all = []
    error_count = 0

    for i in range(target.shape[0]):
        gt = target[0]
        pred = prediction[0]

        try:
            mtx1, mtx2, disparity = procrustes(gt, pred)
            gt_all.append(mtx1)
            pred_all.append(mtx2)
        except Exception as e:
            error_count += 1
            continue

    gt_all = np.array(gt_all)
    pred_all = np.array(pred_all)

    return gt_all, pred_all, error_count


def align_procrustes_old(target, prediction):
    """
    Procrustes MPJPE: MPJPE after rigid alignment (scale, rotation, and translation).
    :param target: Ground truth 3D joint positions, shape [sample, joint, 3]
    :param prediction: Predicted 3D joint positions, shape [sample, joint, 3]
    :return gt_all: Ground truth 3D joint positions after alignment, shape [sample, joint, 3]
    :return pred_all: Predicted 3D joint positions after alignment, shape [sample, joint, 3]
    :return error_count: Error count of failed alignments
    """
    gt_all, pred_all = [], []
    error_count = 0
    joint_number = target.shape[1]

    for gt, pred in zip(target, prediction):
        gt_raw = gt
        if np.sum(np.abs(pred)) != 0:
            transposed = False
            if pred.shape[0] != 3 and pred.shape[0] != 2:
                pred = pred.T
                gt = gt.T
                transposed = True
            assert (gt.shape[1] == pred.shape[1]), "The number of joints must match."

            try:
                muX = np.mean(pred, axis=1, keepdims=True)
                muY = np.mean(gt, axis=1, keepdims=True)

                X0 = pred - muX
                Y0 = gt - muY

                var1 = np.sum(X0 ** 2)
                K = X0.dot(Y0.T)
                U, s, Vh = np.linalg.svd(K)
                V = Vh.T
                Z = np.eye(U.shape[0])
                Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
                R = V.dot(Z.dot(U.T))
                scale = np.trace(R.dot(K)) / var1
                t = muY - scale * (R.dot(muX))
                pred_hat = scale * R.dot(pred) + t
                if transposed:
                    pred_hat = pred_hat.T
            except np.linalg.LinAlgError:
                error_count += 1
                pred_hat = np.tile(np.mean(gt, axis=0), (joint_number, 1))
        else:
            pred_hat = np.tile(np.mean(gt, axis=0), (joint_number, 1))

        gt_all.append(gt_raw)
        pred_all.append(pred_hat)

    gt_all = np.array(gt_all)
    pred_all = np.array(pred_all)

    if error_count > 0:
        print(f"Procrustes alignment failed {error_count} times")

    return gt_all, pred_all, error_count


class Normalize:
    """
    Normalization class to handle scaling of the data. Unnecessary complicated. Based on an old normalization method for
    the training of the morphing model, where the scaling factors needed to be saved.
    """
    def __init__(self):
        self.dict = {}

    def add_key(self, key, mins, maxs):
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
        if vector.size == 0:
            return
        self.add_key(key, np.min(vector), np.max(vector))

    def scale(self, vector, key):
        mins, maxs = self.dict[key]
        return (vector - mins) / (maxs - mins)

    def descale(self, vector, key):
        return vector * (self.dict[key][1] - self.dict[key][0]) + self.dict[key][0]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.dict, f)

    @staticmethod
    def load(path):
        normalize = Normalize()
        with open(path, 'rb') as f:
            normalize.dict = pickle.load(f)
        return normalize


class Framework:
    def __init__(self, model_name, model_type, sample_rate, directory, sweep_id=None):
        # Access the parsed arguments
        self.joint_names = None
        self.model_name = model_name
        self.model_type = model_type
        self.sample_rate = sample_rate
        self.dataset_path = directory
        self.sweep_id = sweep_id

        # Initialize some functions
        self.model_skel_morph = self.load_morph_model()
        self.cam_desynchronizer = frameAug.CameraDesynchronizer()
        self.scaler = Normalize()

        # Initialize empty global variables
        self.sweep_config = None

        # Participants of dataset used in pipeline
        morph = [5, 6, 12, 15, 16, 18, 20, 21, 22, 24, 25]

        self.participants = ['par4', 'par19', 'par11', 'par23']

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

        # self.body_segments = {
        #     "left_lower_arm": ['left_elbow', 'left_wrist'],
        #     "left_upper_arm": ['left_shoulder', 'left_elbow'],
        #     "right_lower_arm": ['right_elbow', 'wrist_right'],
        #     "right_upper_arm": ['shoulder_right', 'elbow_right'],
        #     "left_upper_leg": ['hip_left', 'knee_left'],
        #     "left_lower_leg": ['knee_left', 'ankle_left'],
        #     "right_upper_leg": ['hip_right', 'knee_right'],
        #     "right_lower_leg": ['knee_right', 'ankle_right'],
        #     "left_foot": ['ankle_left', 'toe_left'],
        #     "right_foot": ['ankle_right', 'toe_right']
        # }

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
        morph_model_path = f"skeletonMorphing/models/model_skeleton_morph_{self.model_name}.pth"
        model_skel_morph = modelSkeletonMorphing.Synthesizer(dropout_rate=0)
        if os.path.isfile(morph_model_path):
            model_skel_morph.load_state_dict(torch.load(morph_model_path))
            model_skel_morph.eval()
        else:
            raise ValueError('Morphing model is missing')

        # Ensure the model is on the correct device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_skel_morph = model_skel_morph.to(device)

        return model_skel_morph

    def normalize_input(self, data, key):
        return self.scaler.scale(data, key)

    def denormalize_output(self, data, key):
        return self.scaler.descale(data, key)

    def apply_morphing(self, input_pose):
        """
        Apply skeleton morphing to the input pose using the loaded model.
        :param input_pose: Input pose to be morphed as numpy array.
        :return: morphed_pose: Pose after applying skeleton morphing.
        """
        # Normalize the input data
        input_pose = self.normalize_input(input_pose, "pose_gt")

        # Convert input pose to tensor and apply morphing
        input_pose = torch.from_numpy(input_pose)
        device = next(self.model_skel_morph.parameters()).device
        inp_poses = input_pose.view(-1, input_pose.size(1) * input_pose.size(2)).float().to(device)

        with torch.no_grad():
            morphed_pose = self.model_skel_morph(inp_poses)

        # Denormalize the output data
        morphed_pose = morphed_pose.view(-1, input_pose.size(1), input_pose.size(2)).cpu().detach().numpy()
        morphed_pose = self.denormalize_output(morphed_pose, 'pose_gt')
        return morphed_pose

    def convert_keypoints_dicts_to_array(self, keypoints_all):
        """
        Convert a list of dictionaries containing keypoints to a list of numpy arrays.
        :param keypoints_all: List of dictionaries where each dictionary contains keypoints.
        :return: all_keypoints_array: List of numpy arrays with keypoints.
        """
        all_keypoints_array = []
        for keypoints_dict in keypoints_all:
            keypoints_list = [keypoints_dict[key] for key in sorted(keypoints_dict.keys())]
            keypoints_array = np.stack(keypoints_list, axis=1)
            all_keypoints_array.append(keypoints_array)
        return all_keypoints_array

    def calculate_log_metrics(self, gt_keypoints, pred_keypoints, inference_times):
        """
        Calculate metrics for each sweep and logs them on wandb. Calculates it for the whole body and body segments.
        :param gt_keypoints: [[x0,y0,z0], [x1,y1,z1], [x2,y2,z2], ...]
        :param pred_keypoints: [[x0,y0,z0], [x1,y1,z1], [x2,y2,z2], ...]
        :param inference_times: [inference_time0, inference_time1, inference_time2, ]
        """
        gt_keypoints_all_arrays = self.convert_keypoints_dicts_to_array(gt_keypoints)
        pred_keypoints_all_arrays = self.convert_keypoints_dicts_to_array(pred_keypoints)
        joint_names = list(gt_keypoints[0].keys())
        assert joint_names == list(gt_keypoints[0].keys()), "Joint names of target and prediction do not match."

        metrics_dict = {metric: {joint: [] for joint in joint_names} for metric in
                        ["pmpjpe_m", "pmpjpe_s", "velocity_m", "velocity_s", "pcc", "pvalue"]}
        all_metrics_dict = {metric: [] for metric in
                            ["pmpjpe_m", "pmpjpe_s", "velocity_m", "velocity_s", "angular_m", "angular_s", "r2",
                             "pcc", "pvalue"]}

        # Calculate metrics for each movement type, to avoid jumps between movement segments
        for gt_movement, pred_movement in zip(gt_keypoints_all_arrays, pred_keypoints_all_arrays):
            # Calculate metrics for each joint
            for joint in range(np.size(gt_movement, 1)):
                gt_array = np.squeeze(np.array(gt_movement[:, joint, :]))
                pred_array = np.squeeze(np.array(pred_movement[:, joint, :]))

                pmpjpe_m, pmpjpe_s = metrics.calculate_mpjpe(gt_array, pred_array)
                velocity_m, velocity_s = metrics.mean_velocity_error(gt_array, pred_array, self.sample_rate)
                pcc, pvalue = metrics.calculate_correlation(gt_array, pred_array)

                metrics_dict["pmpjpe_m"][joint_names[joint]].append(pmpjpe_m)
                metrics_dict["pmpjpe_s"][joint_names[joint]].append(pmpjpe_s)
                metrics_dict["velocity_m"][joint_names[joint]].append(velocity_m)
                metrics_dict["velocity_s"][joint_names[joint]].append(velocity_s)
                metrics_dict["pcc"][joint_names[joint]].append(pcc)
                metrics_dict["pvalue"][joint_names[joint]].append(pvalue)

            # Calculate whole body metrics using all keypoints
            all_metrics_dict["pmpjpe_m"].append(
                np.mean([np.mean(metrics_dict["pmpjpe_m"][joint]) for joint in joint_names]))
            all_metrics_dict["pmpjpe_s"].append(
                np.mean([np.mean(metrics_dict["pmpjpe_s"][joint]) for joint in joint_names]))
            all_metrics_dict["velocity_m"].append(
                np.mean([np.mean(metrics_dict["velocity_m"][joint]) for joint in joint_names]))
            all_metrics_dict["velocity_s"].append(
                np.mean([np.mean(metrics_dict["velocity_s"][joint]) for joint in joint_names]))
            all_metrics_dict["pcc"].append(
                np.mean([np.mean(metrics_dict["pcc"][joint]) for joint in joint_names]))
            all_metrics_dict["pvalue"].append(
                np.mean([np.mean(metrics_dict["pvalue"][joint]) for joint in joint_names]))

        gt_keypoints_all_arrays = np.concatenate(gt_keypoints_all_arrays, axis=0)
        pred_keypoints_all_arrays = np.concatenate(pred_keypoints_all_arrays, axis=0)
        gt_movement_dict = {joint: gt_keypoints_all_arrays[:, joint_names.index(joint)] for joint in joint_names}
        pred_movement_dict = {joint: pred_keypoints_all_arrays[:, joint_names.index(joint)] for joint in joint_names}

        angular_m, angular_s, r2 = metrics.calculate_angle_error(gt_movement_dict, pred_movement_dict)
        all_metrics_dict["angular_m"].append(angular_m)
        all_metrics_dict["angular_s"].append(angular_s)
        all_metrics_dict["r2"].append(r2)

        # Log metrics for each joint
        for metric, values in metrics_dict.items():
            wandb.log({f"{metric}_joint": {joint: np.mean(vals) for joint, vals in values.items()}})

        # Flatten inference_times into a single list
        flattened_inference_times = [time for sequence in inference_times for time in sequence]

        # Log each metric from all_metrics_dict
        for metric, values in all_metrics_dict.items():
            if isinstance(values, list):
                if all(isinstance(v, dict) for v in values):
                    # Handle list of dictionaries (e.g., angular_m, angular_s, r2)
                    combined_dict = {}
                    for sub_dict in values:
                        for sub_metric, sub_value in sub_dict.items():
                            if sub_metric not in combined_dict:
                                combined_dict[sub_metric] = []
                            combined_dict[sub_metric].append(sub_value)
                    for sub_metric, sub_values in combined_dict.items():
                        combined_mean = np.mean(sub_values)
                        wandb.log({f"{metric}_{sub_metric}": combined_mean})

                    # Calculate overall mean and std for angular and r2 metrics
                    if metric in ['angular_m', 'angular_s', 'r2']:
                        all_values = np.concatenate(
                            [np.array(sub_values).reshape(-1) for sub_values in combined_dict.values()])
                        overall_mean = np.mean(all_values)
                        overall_std = np.std(all_values)
                        wandb.log({f"{metric}_overall_mean": overall_mean, f"{metric}_overall_std": overall_std})

                elif all(isinstance(v, (list, np.ndarray)) for v in values):
                    # Handle list of arrays
                    combined_array = np.concatenate(values)
                    wandb.log({f"{metric}": wandb.Histogram(np_histogram=np.histogram(combined_array))})
                else:
                    # Handle list of numerical values (e.g., pmpjpe_m, pmpjpe_s, velocity_m, velocity_s, pcc, pvalue)
                    combined_values = np.concatenate([np.array(v).reshape(-1) for v in values])
                    combined_mean = np.mean(combined_values)
                    wandb.log({metric: combined_mean})
            else:
                # Handle individual values
                wandb.log({metric: values})

        # Log inference time
        inference_time_mean = np.mean(np.array(flattened_inference_times))
        inference_time_std = np.std(np.array(flattened_inference_times))
        wandb.log({"inference_time_mean": inference_time_mean, "inference_time_std": inference_time_std})

    def main(self, config=None):
        """
        Run inference on the chosen model with sweep parameters and log results to wandb project.
        """
        with wandb.init(config=config) as run:
            # If called by wandb.agent, as below, this config will be set by Sweep Controller
            config = wandb.config
            run.name = f"{config.model_name}-{config.movement}-{config.augmentation}"

            # Determine cameras based on augmentation type
            if 'cameras_' in config.augmentation:
                cameras = list(map(int, config.augmentation.replace('cameras_', '').split('_')))
            else:
                cameras = config.default_camera

            wandb.log({'cameras': cameras})

            # Generate cv2 video API and load respective ground truth keypoints
            gt_keypoints_all = []
            pred_keypoints_all = []
            inference_times_all = []
            error_count_all = 0

            for par in tqdm(self.participants, ascii=True, desc="Participant:"):
                for mov in tqdm(self.movement_category[config.movement], ascii=True, desc="Movement:"):

                    gt_keypoints, caps = readDataEval.load_data(self.dataset_path, par, mov, cameras, self.model_name)
                    if not gt_keypoints or not caps:
                        print(f"Participant {par} and movement {mov} not found")
                        continue

                    # Open model based on name and run inference
                    if self.model_type == "mono":
                        # Just load the first (and only) camera
                        gt_keypoints = gt_keypoints[0][1]
                        caps = caps[0][1]
                        if self.model_name == "mediapipe":
                            pred_keypoints, inference_times, frame = mediapipeMono.inference_video(caps, config)
                            selected_columns = [12, 11, 14, 13, 16, 15, 24, 23, 26, 25, 28, 27, 30, 29, 32,
                                                31]  # Select only relevant keypoints and put them in the right order
                            pred_keypoints = pred_keypoints[:, selected_columns, 1:]

                            # Joint names mapping for MediaPipe (kinda redundant, as names are included in gt)
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
                            p_matrix = camCali.get_projection_matrix(cameras, self.sweep_config['decalibration'])
                        if self.model_name == "openpose":
                            print("OpenPose not implemented yet")
                            # pred_keypoints, inference_times = canonPoseMulti.inference_video(caps)
                        if self.model_name == "LWCDR":
                            print("LWCDR not implemented yet")
                            # pred_keypoints, inference_times = canonPoseMulti.inference_video(caps)

                    # Add min and max values for normalization of pose_gt
                    gt_keypoints_np = np.array(gt_keypoints)
                    self.scaler.add_key_from_vector(gt_keypoints_np, "pose_gt")

                    # Procrustes Alignment
                    gt_keypoints, pred_keypoints, error_count = align_procrustes_old(gt_keypoints, pred_keypoints)
                    error_count_all += error_count

                    # Morph ground truth to format of predicted keypoints (can't handle gaps)
                    pred_keypoints = self.apply_morphing(pred_keypoints)

                    # Postprocess predicted keypoints
                    pred_keypoints = postprocessing.postprocess_points(pred_keypoints,
                                                                       self.interpolation_fun,
                                                                       self.smoothing_fun)

                    # Add joint names
                    pred_keypoints = {self.joint_names[i]: pred_keypoints[:, i, :] for i in
                                      range(pred_keypoints.shape[1])}

                    gt_keypoints = {self.joint_names_gt[i]: gt_keypoints[:, i, :] for i in
                                    range(gt_keypoints.shape[1])}

                    # Collect pred_keypoints for each movement iteration
                    gt_keypoints_all.append(gt_keypoints)
                    pred_keypoints_all.append(pred_keypoints)
                    inference_times_all.append(inference_times)

                    # Collect last good frame and keypoints for visualization
                    if frame is not None or frame.size != 0:
                        if gt_keypoints is not None and pred_keypoints is not None:
                            frame_example = frame
                            gt_keypoints_example = {key: value[-1] for key, value in gt_keypoints.items()}
                            pred_keypoints_example = {key: value[-1] for key, value in pred_keypoints.items()}

                            data = {'gt': gt_keypoints_example,
                                    'pred': pred_keypoints_example,
                                    'frame': frame_example}
                            with open('results/' + run.name + str(mov) + '.pkl', 'wb') as file:
                                pickle.dump(data, file)

            # Save data for debugging
            data_to_save = {'gt_keypoints': gt_keypoints_all,
                            'pred_keypoints': pred_keypoints_all,
                            'inference_time': inference_times_all,
                            'frame': frame
                            }

            # Save calculate keypoints for later
            with open('results/' + run.name + '.pkl', 'wb') as file:
                pickle.dump(data_to_save, file)

            print("Calculating metrics and logging to wandb")
            self.calculate_log_metrics(gt_keypoints_all, pred_keypoints_all, inference_times_all)
            plot_3d_keypoints_validation(gt_keypoints_example, pred_keypoints_example, self.model_name)
            log_frame_example(frame_example)
            samples = self.total_frames(gt_keypoints_all)
            wandb.log({"sample_number": samples})
            wandb.log({"procrustes_number": samples - error_count_all})

    def total_frames(self, data):
        key = 'right_shoulder'  # Random keypoint to calculate number of frames
        total = 0
        for item in data:
            if key in item:
                total += item[key].shape[0]
        return total

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
                raise ValueError('Choose a valid mono-ocular model, or change the model type to multi')
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
                    'values': ['none', 'defocus', 'underexposure', 'occlusion', 'background',
                               'cameras_0', 'cameras_1', 'cameras_2', 'cameras_3', 'cameras_4', 'cameras_5']
                },
                'default_camera': {
                    'value': [5]
                }
            }
        elif self.model_type == 'multi':
            # Sweep parameters
            parameters_dict = {
                'augmentation': {
                    'values': ['none', 'defocus', 'underexposure', 'occlusion', 'background',
                               'desynchronize', 'decalibration',
                               'cameras_4_0', 'cameras_5_1', 'cameras_4_3', 'cameras_4_2',
                               'cameras_0_4_3', 'cameras_5_4_1', 'cameras_0_4_3_2']
                },
                'default_camera': {
                    'value': [4, 0]
                }
            }

        self.sweep_config['parameters'].update(parameters_dict)

        if not self.sweep_id:
            self.sweep_id = wandb.sweep(sweep=self.sweep_config, project='HPE_framework')
            print(f"Sweep initialized with ID: {self.sweep_id}")
        else:
            print(f"Sweep already initialized with ID: {self.sweep_id}")
        # Training complete
        print("Testing complete")

    def run_sweep_agent(self):
        print(f"Starting wandb agent for sweep ID: {self.sweep_id}")
        wandb.agent(self.sweep_id, function=self.main)


# Run the framework
framework = Framework(model_name="mediapipe", model_type="mono", sample_rate=25,
                      directory="/media/emanu/Emanuel Lorenz1/MoCap/segmented", sweep_id=None)
framework.initiate_wandb_sweep()
framework.run_sweep_agent()

# To start a new run enter this into a console:
# python /home/emanu/Documents/PycharmProjects/HPEClinicalEvaluation/main_pipeline_debug.py

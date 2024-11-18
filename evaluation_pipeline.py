import os
import sys
import wandb
import numpy as np
import yaml
import torch
import cv2
cv2.setUseOptimized(True)
cv2.setNumThreads(2)
from tqdm import tqdm
from skeletonMorphing import modelSkeletonMorphing
from utils import cameraCalibration as camCali
from utils import frameAugmentation as frameAug
from utils.plotKeypoints import plot_3d_keypoints_validation
from utils import metrics, postprocessing, readDataEval
from models import mediapipeMono
from models import mediapipeMulti
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

    wandb.log({"Example_Frame": wandb.Image(frame)})


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
    gt_all, pred_all, error_count = [], [], 0

    for gt, pred in zip(target, prediction):
        try:
            mtx1, mtx2, disparity = procrustes(gt, pred)
            gt_all.append(mtx1)
            pred_all.append(mtx2)
        except Exception:
            error_count += 1
            continue

    return np.array(gt_all), np.array(pred_all), error_count


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
                Z[-1, -1] *= np.sign(np.linalg.det(U.dot(Vh)))
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



class Framework:
    def __init__(self, model_name, model_type, sample_rate, directory, sweep_id=None):
        # Access the parsed arguments
        self.model_name = model_name
        self.model_type = model_type
        self.sample_rate = sample_rate
        self.dataset_path = directory
        self.sweep_id = sweep_id

        # Initialize some functions
        self.model_skel_morph = self.load_morph_model()
        self.cam_desynchronizer = frameAug.CameraDesynchronizer()

        self.participants = ['par4', 'par19', 'par11', 'par23']
        # self.participants = ['par4']

        # Defines movement number in dataset related to different movement categories
        self.movement_category = {
            "upper": [1, 2, 3, 4],
            "lower": [5, 6, 7, 8],
            "complex": [9, 10, 11, 12, 13],
            "sitting": [14, 15, 16, 17]
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
        model_path = f"skeletonMorphing/models/model_skeleton_morph_{self.model_name}_final.pth"
        model_skel_morph = modelSkeletonMorphing.Synthesizer(dropout_rate=0, layer_size=1024)
        if not os.path.isfile(model_path):
            raise ValueError(f'Morphing model not found at {model_path}')

        model_skel_morph.load_state_dict(torch.load(model_path))
        model_skel_morph.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return model_skel_morph.to(device)


    def apply_morphing(self, input_pose):
        """
        Apply skeleton morphing to the input pose using the loaded model.
        :param input_pose: Input pose to be morphed as numpy array.
        :return: morphed_pose: Pose after applying skeleton morphing.
        """
        # Convert input pose to tensor and apply morphing
        input_pose = torch.from_numpy(input_pose).float()
        device = next(self.model_skel_morph.parameters()).device
        inp_poses = input_pose.view(-1, input_pose.size(1) * input_pose.size(2)).float().to(device)

        with torch.no_grad():
            morphed_pose = self.model_skel_morph(inp_poses)

        # Denormalize the output data
        morphed_pose = morphed_pose.view(-1, input_pose.size(1), input_pose.size(2)).cpu().detach().numpy()
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

        if len(gt_keypoints_all_arrays) != len(pred_keypoints_all_arrays):
            raise ValueError("Ground truth and prediction arrays have different lengths!")

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

    def evaluation(self, config=None):
        """
        Run inference on the chosen model with sweep parameters and log results to wandb project.
        """
        with wandb.init(config=config) as run:
            config = wandb.config
            run.name = f"{config.model_name}-{config.movement}-{config.augmentation}"
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

                    gt_keypoints, caps = readDataEval.load_data(self.dataset_path, par, mov, cameras)
                    if not gt_keypoints or not caps:
                        print(f"Participant {par} and movement {mov} not found")
                        continue

                    # Monocular models
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
                            self.joint_num_total = 33

                    # Multioccular models
                    elif self.model_type == "multi":
                        if self.model_name == 'mediapipe':
                            gt_keypoints = gt_keypoints[0][1]
                            self.joint_num_total = 33

                        # Desynchronize video streams
                        if config._items['augmentation'] == 'desynchronize':
                            caps = self.cam_desynchronizer.desynchronize(caps)

                        # Load camera parameter matrix and add noise if specified so
                        if config._items['augmentation'] == 'decalibration':
                            p_matrix_raw, _, _ = camCali.get_projection_matrix(cameras, True)
                        else:
                            p_matrix_raw, _, _ = camCali.get_projection_matrix(cameras, False)

                        p_matrix = list(p_matrix_raw.values())

                        if self.model_name == "mediapipe":
                            pred_keypoints, inference_times, frame = mediapipeMulti.inference_video(caps, p_matrix, config)
                            selected_columns = [12, 11, 14, 13, 16, 15, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31]  # Select only relevant keypoints and put them in the right order
                            pred_keypoints = pred_keypoints[:, selected_columns, :]

                            # Joint names mapping for MediaPipe (kinda redundant, as names are included in gt)
                            self.joint_names = {
                                0: 'right_shoulder', 1: 'left_shoulder', 2: 'right_elbow', 3: 'left_elbow',
                                4: 'right_wrist', 5: 'left_wrist', 6: 'right_hip', 7: 'left_hip',
                                8: 'right_knee', 9: 'left_knee', 10: 'right_ankle', 11: 'left_ankle',
                                12: 'right_heel', 13: 'left_heel', 14: 'right_foot_index', 15: 'left_foot_index'
                            }

                    gt_keypoints_np = np.array(gt_keypoints)
                    if gt_keypoints_np.shape != pred_keypoints.shape:
                        print(gt_keypoints.shape)
                        print(pred_keypoints.shape)
                        assert gt_keypoints_np.shape == pred_keypoints.shape, "Not the same shape. Bummer!"

                    # Postprocess predicted keypoints
                    #pred_keypoints = postprocessing.postprocess_points(pred_keypoints,
                    #                                                   self.interpolation_fun,
                    #                                                   self.smoothing_fun)

                    # Procrustes Alignment
                    gt_keypoints, pred_keypoints, error_count = align_procrustes_old(gt_keypoints_np, pred_keypoints)
                    error_count_all += error_count

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
            samples_gt = self.total_frames(gt_keypoints_all)
            samples_pred = self.total_frames(pred_keypoints_all)
            wandb.log({"sample_number_gt": samples_gt})
            wandb.log({"sample_number_pred": samples_pred})
            wandb.log({"procrustes_number": samples_gt - error_count_all})


    def total_frames(self, data):
        key = 'right_shoulder'  # Random keypoint to calculate number of frames
        total = 0
        for item in data:
            if key in item:
                total += item[key].shape[0]
        return total


# Sweep function
def sweep(model_type):
    """
    Initializes a W&B sweep based on the model type and configuration file.
    :param model_type: The type of model, either 'mono' or 'multi'.
    """
    # Define configuration file paths
    config_files = {
        'mono': 'config_mono.yaml',
        'multi': 'config_multi.yaml'
    }
    if model_type not in config_files:
        raise ValueError("Invalid model_type. Use 'mono' or 'multi'.")

    # Load configuration from the appropriate file
    try:
        with open(config_files[model_type], 'r') as file:
            config = yaml.safe_load(file)
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise ValueError(f"Error loading configuration file: {e}")

    # Initialize W&B sweep
    run = wandb.init(config=config)
    print(config)

    framework = Framework(
        model_name=config['parameters']['model_name']['value'],
        model_type=config['parameters']['model_type']['value'],
        sample_rate=config['parameters']['sample_rate']['value'],
        directory=config['parameters']['dataset_path']['value'],
    )
    framework.evaluation(config)
    wandb.finish()


if __name__ == "__main__":
    sweep(sys.argv[1])
    #r"C:\Users\vizlab_stud\emanuel"
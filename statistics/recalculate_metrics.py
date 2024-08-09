import os
import numpy as np
import pickle
from utils import metrics, postprocessing, readDataEval


def convert_to_array(joint_dict):
    """
    Helper-function to convert a dictionary of joint keypoints to a numpy array with shape [joints, 3].
    :param joint_dict: Dictionary where keys are joint names and values are 3D coordinates.
    :return: Numpy array with shape [joints, 3].
    """
    return np.array(list(joint_dict.values())), list(joint_dict.keys())

def calculate_joint_correlations(target_array, pred_array):
    joint_correlations = []
    joint_pvalues = []

    num_keypoints = target_array.shape[2]  # Assuming (samples, XYZ, keypoints)

    for joint in range(num_keypoints):
        target = target_array[:, :, joint]
        prediction = pred_array[:, :, joint]

        # Calculate correlation using the provided function
        correlation, pvalue = metrics.calculate_correlation(target, prediction)
        joint_correlations.append(correlation)
        joint_pvalues.append(pvalue)

    return np.mean(joint_correlations), np.mean(joint_pvalues)

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


def group_files_by_suffix(files):
    """
    Group files by the suffix after the last '-'.
    :param files: List of filenames.
    :return: Dictionary where keys are suffixes and values are lists of files with that suffix.
    """
    from collections import defaultdict
    grouped_files = defaultdict(list)
    for file in files:
        suffix = file.split('-')[-1]
        grouped_files[suffix].append(file)
    return grouped_files


def process_and_calculate_metrics_for_groups(grouped_files, directory):
    """
    Process each group of files and calculate metrics.
    :param grouped_files: Dictionary of grouped files by suffix.
    :param directory: Directory containing the files.
    :return: Dictionary with calculated metrics for each group.
    """
    all_metrics = {}
    keypoints_metrics = {}

    for suffix, files in grouped_files.items():
        print(f"Calculating metrics for conditions with suffix: {suffix}")
        combined_pred_keypoints = []
        combined_gt_keypoints = []
        combined_infer_time = []


        # Read and combine data from all files in the group
        for file in files:
            with open(os.path.join(directory, file), 'rb') as f:
                data = pickle.load(f)
                # Convert the data before appending
                # Convert the keypoints to array and get names
                pred_values, pred_names = zip(*[convert_to_array(d) for d in data['pred_keypoints']])
                gt_values, gt_names = zip(*[convert_to_array(d) for d in data['gt_keypoints']])

                # Extend the combined lists with new data
                combined_pred_keypoints.extend(pred_values)
                combined_gt_keypoints.extend(gt_values)
                combined_infer_time.extend(data['inference_time'])

        # Convert to numpy arrays for processing
        combined_pred_keypoints = np.concatenate(combined_pred_keypoints, axis=1)
        combined_gt_keypoints = np.concatenate(combined_gt_keypoints, axis=1)
        combined_infer_time = np.concatenate(combined_infer_time, axis=0)

        # Align keypoints using Procrustes alignment
        aligned_gt, aligned_pred, error_count = align_procrustes_old(combined_gt_keypoints, combined_pred_keypoints)

        # Swap axes to match with the expected input from metrics
        aligned_gt = np.moveaxis(aligned_gt, [0, 1, 2], [2, 0, 1])
        aligned_pred = np.moveaxis(aligned_pred, [0, 1, 2], [2, 0, 1])

        # Calculate metrics (adapt this part based on the metrics you want)
        pmpjpe_m, pmpjpe_s = metrics.calculate_mpjpe(aligned_gt, aligned_pred)
        velocity_m, velocity_s = metrics.mean_velocity_error(aligned_gt, aligned_pred, sample_rate=25)
        pcc, pvalue = calculate_joint_correlations(aligned_gt, aligned_pred)

        # Store the metrics
        all_metrics[suffix] = {
            'pmpjpe_m': pmpjpe_m,
            'pmpjpe_s': pmpjpe_s,
            'velocity_m': velocity_m,
            'velocity_s': velocity_s,
            'pcc': pcc,
            'pvalue': pvalue,
            #'angular_m': angular_m,
            #'angular_s': angular_s,
            'inference_time_mean': np.mean(combined_infer_time),
            'inference_time_std': np.std(combined_infer_time),
            'sample_num': np.size(aligned_gt, 1)
        }

        # Initialize dictionary for keypoint-specific metrics
        keypoints_metrics[suffix] = {}

        # Calculate and store metrics for each keypoint
        num_keypoints = aligned_gt.shape[2]
        for i in range(num_keypoints):
            gt_keypoint = aligned_gt[:, :, i]
            pred_keypoint = aligned_pred[:, :, i]

            # Calculate metrics for this keypoint
            pmpjpe_m_kp, pmpjpe_s_kp = metrics.calculate_mpjpe(gt_keypoint, pred_keypoint)
            velocity_m_kp, velocity_s_kp = metrics.mean_velocity_error(gt_keypoint, pred_keypoint, sample_rate=25)
            pcc_kp, pvalue_kp = metrics.calculate_correlation(gt_keypoint, pred_keypoint)

            # Store keypoint-specific metrics
            keypoints_metrics[suffix][gt_names[0][i]] = {
                'pmpjpe_m': pmpjpe_m_kp,
                'pmpjpe_s': pmpjpe_s_kp,
                'velocity_m': velocity_m_kp,
                'velocity_s': velocity_s_kp,
                #'angular_m': angular_m,
                #'angular_s': angular_s,
                'pcc': pcc_kp,
                'pvalue': pvalue_kp,
                'sample_num': np.size(gt_keypoint, 1)
            }

    return all_metrics, keypoints_metrics


def main():
    """
    Run inference on the chosen model with sweep parameters and log results to wandb project.
    """
    directory = '/home/emanu/Desktop/mono'

    # List of all files in the directory
    all_files = os.listdir(directory)

    # Group files by suffix
    grouped_files = group_files_by_suffix(all_files)

    # Process each group and calculate metrics
    all_metrics, keypoints_metrics = process_and_calculate_metrics_for_groups(grouped_files, directory)

    # Now, metrics_results contains metrics for each group, you can log them to wandb or print them
    for suffix, metrics in all_metrics.items():
        print(f"Overall metrics for group {suffix}:")
        print(metrics)

    for suffix, metrics in keypoints_metrics.items():
        print(f"Keypoint-specific metrics for group {suffix}:")
        print(metrics)


main()
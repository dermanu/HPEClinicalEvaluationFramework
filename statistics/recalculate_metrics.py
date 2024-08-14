import os
import numpy as np
import pickle
from utils import metrics
from scipy import stats
import statsmodels.stats.api as sms
import multiprocessing
from cliffs_delta import cliffs_delta
import pandas as pd
from utils import angle_metrics_tpose
import matplotlib.pyplot as plt


def sliding_window(arr, window_size, step_size):
    """
    Generates sliding windows over the temporal axis of a 3D array.

    :param arr: 3D array with shape [time, joint, 3]
    :param window_size: Size of the sliding window
    :param step_size: Step size between windows
    :return: Generator yielding windows of the original array
    """
    for start in range(0, arr.shape[0] - window_size + 1, step_size):
        yield arr[start:start + window_size]


def worker_mpjpe(inputs):
    aligned_gt_sample, aligned_pred_sample = inputs
    return metrics.calculate_mpjpe(aligned_gt_sample[np.newaxis, :, :], aligned_pred_sample[np.newaxis, :, :])


def worker_velocity_and_corr(inputs):
    gt_window, pred_window, sample_rate = inputs
    velocity = metrics.mean_velocity_error(gt_window, pred_window, sample_rate=sample_rate)
    corr, _ = calculate_joint_correlations(gt_window, pred_window)
    return velocity, corr


def calculate_metrics_for_sample(aligned_gt, aligned_pred, sample_rate):
    # Parameters for the sliding window
    window_size = 26  # Adjust based on the temporal granularity you need
    step_size = 13  # Adjust for overlap between windows

    # Setup multiprocessing pool
    pool = multiprocessing.Pool(processes=18)

    # Calculate MPJPE in parallel
    mpjpe_inputs = [(aligned_gt[j], aligned_pred[j]) for j in range(np.size(aligned_gt, 0))]
    pmpjpe = pool.map(worker_mpjpe, mpjpe_inputs)

    # Prepare inputs for velocity and correlation calculations
    velocity_and_corr_inputs = [
        (gt, pred, sample_rate) for gt, pred in zip(
            sliding_window(aligned_gt, window_size, step_size),
            sliding_window(aligned_pred, window_size, step_size)
        )
    ]

    # Calculate velocity and correlation in parallel
    velocity_and_corr_results = pool.map(worker_velocity_and_corr, velocity_and_corr_inputs)
    velocity = [result[0] for result in velocity_and_corr_results]
    pcc = [result[1] for result in velocity_and_corr_results]

    # Close the pool and wait for work to finish
    pool.close()
    pool.join()

    return pmpjpe, velocity, pcc


def compare_metrics_with_none_group(data, alpha=0.05):
    """
    Compare different data augmentation methods to the original data,
    including normality check to decide between t-test and Wilcoxon test.

    Parameters:
    - data_original: numpy array of original data metrics
    - data_augmented: list of numpy arrays, each corresponding to metrics from an augmented dataset
    - metrics: list of strings with the names of the metrics being tested
    - alpha: significance level for the normality test

    Returns:
    - results: dictionary with p-values, confidence intervals, and effect sizes, and differences
     for each metric and augmentation
    """
    baseline = data['none.pkl']
    other_conditions = {k: v for k, v in data.items() if k != 'none.pkl'}
    metrics = baseline.keys()
    results = {}

    for i, metric in enumerate(metrics):
        results[metric] = []
        p_values = []
        test_results = []
        base_data = np.array(baseline[metric])
        if len(base_data.shape) > 1:
            base_data = base_data[:, 0]
        # Check for normality using Shapiro-Wilk test. Can't handle large samples.
        stat_orig, p_value_orig = stats.shapiro(np.random.choice(base_data, size=1000, replace=False))

        for condition_name, condition_data in other_conditions.items():
            aug_data = np.array(condition_data[metric])
            if len(aug_data.shape) > 1:
                aug_data = aug_data[:, 0]
            # Check for normality using Shapiro-Wilk test. Can't handle large samples.
            stat_aug, p_value_aug = stats.shapiro(np.random.choice(aug_data, size=1000, replace=False))

            # Calculate the difference in percent and unit
            base_mean = np.mean(base_data)
            aug_mean = np.mean(aug_data)
            diff_percent = ((base_mean - aug_mean)/base_mean)*100
            diff_total = aug_mean - base_mean

            # As the different conditions are independent of each others
            # (focus on artifacts, not on same participants)
            if p_value_orig > alpha and p_value_aug > alpha:
                # Data is normally distributed, use t-test
                print('The data is normal distributed -> t-test test')
                t_stat, p_value = stats.ttest_rel(aug_data, base_data)
                test_used = 't-test'
            else:
                # Data is not normally distributed, use Mann-Whitney U test
                print('The data is not normal distributed -> Wilcoxon')
                t_stat, p_value = stats.wilcoxon(aug_data, base_data)
                test_used = 'Wilcoxon'

            p_values.append(p_value)

            # Calculate effect size (Cohen's d or Cliff's delta)
            if test_used == 't-test':
                effect_size = (np.mean(aug_data) - np.mean(base_data)) / (np.sqrt((np.stdev(aug_data) ** 2 + np.stdev(base_data) ** 2) / 2))

                if np.abs(effect_size) < 0.2:
                    effect_interpret = 'negligible'
                elif np.abs(effect_size) >= 0.2 & np.abs(effect_size) < 0.5:
                    effect_interpret = 'small'
                elif np.abs(effect_size) >= 0.5 & np.abs(effect_size) < 0.8:
                    effect_interpret = 'medium'
                else:
                    effect_size = 'large'
            else:
                effect_size = t_stat / np.sqrt(np.size(aug_data))

                if np.abs(effect_size) < 0.1:
                    effect_interpret = 'negligible'
                elif np.abs(effect_size) >= 0.1 & np.abs(effect_size) < 0.3:
                    effect_interpret = 'small'
                elif np.abs(effect_size) >= 0.3 & np.abs(effect_size) < 0.5:
                    effect_interpret = 'medium'
                else:
                    effect_size = 'large'

            # Calculate confidence interval (if applicable)
            if test_used == 't-test':
                cm = sms.CompareMeans(sms.DescrStatsW(aug_data), sms.DescrStatsW(base_data))
                conf_int = cm.tconfint_diff(usevar='unequal')  # Use 'unequal' if variances are different
            else:
                conf_int = (None, None)

            if p_value < 0.05:
                significant = True
            else:
                significant = False

            # Append results
            results[metric].append({
                'augmentation': condition_name,
                'test_used': test_used,
                'Significant': significant,
                'p_value': p_value,
                'confidence_interval': conf_int,
                'effect_size': effect_size,
                'effect_interpretation': effect_interpret,
                'difference_percent': diff_percent,
                'difference_total': diff_total
            })

        # Apply Benjamini-Hochberg correction
        p_values_sorted_indices = np.argsort(p_values)
        m = len(p_values)
        corrected_alpha_values = [(i + 1) / m * alpha for i in range(m)]
        bh_significant = [p_value <= corrected_alpha_values[i] for i, p_value in enumerate(np.array(p_values)[p_values_sorted_indices])]

        for i, idx in enumerate(p_values_sorted_indices):
            test_results[idx]['Significant'] = bh_significant[i]
            test_results[idx]['corrected_alpha'] = corrected_alpha_values[i]
            results[metric].append(test_results[idx])


    return results


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

    for file in files:
        if file.split('-')[-1] == 'none.pkl':
            suffix = file.split('-')[-2]
            grouped_files[suffix].append(file)

    return grouped_files


def process_and_calculate_metrics_for_groups(grouped_files, directory):
    """
    Process each group of files, calculate metrics, and return raw keypoints.
    :param grouped_files: Dictionary of grouped files by suffix.
    :param directory: Directory containing the files.
    :return: Dictionary with calculated metrics, keypoint-specific metrics, and raw keypoints for each group.
    """
    all_metrics = {}
    all_metrics_single = {}
    keypoints_metrics = {}
    keypoints_metrics_single = {}

    for suffix, files in grouped_files.items():
        print(f"Calculating metrics for conditions with suffix: {suffix}")
        combined_pred_keypoints = []
        combined_gt_keypoints = []
        combined_infer_time = []

        # Read and combine data from all files in the group
        for file in files:
            with open(os.path.join(directory, file), 'rb') as f:
                data = pickle.load(f)
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
        angles_gt = angle_metrics_tpose.calculate_angles_tpose(aligned_gt)
        angles_pred = angle_metrics_tpose.calculate_angles_tpose(aligned_pred)
        angle_errors = {
            joint: np.mean(np.degrees(np.abs(angles_gt[joint] - angles_pred[joint])), axis=-1) for joint in
            angles_gt.keys()
        }
        all_angle_errors = np.concatenate([angle_errors[joint].flatten() for joint in angle_errors.keys()])
        angle_m = np.mean(all_angle_errors)
        angle_s = np.std(all_angle_errors)
        # Combine the results across all joints by averaging across the frames for all joints for the statistics
        stat_angle_error = np.mean(list(angle_errors.values()), axis=0)

        pmpjpe_m, pmpjpe_s = metrics.calculate_mpjpe(aligned_gt, aligned_pred)
        velocity_m, velocity_s = metrics.mean_velocity_error(aligned_gt, aligned_pred, sample_rate=25)
        pcc, pvalue = calculate_joint_correlations(aligned_gt, aligned_pred)

        # Calculate metrics in parallel using multiprocessing
        pmpjpe, velocity, pcc_single = calculate_metrics_for_sample(aligned_gt, aligned_pred, sample_rate=25)

        # Store the metrics
        all_metrics[suffix] = {
            'pmpjpe_m': pmpjpe_m,
            'pmpjpe_s': pmpjpe_s,
            'angle_m': angle_m,
            'angle_s': angle_s,
            'velocity_m': velocity_m,
            'velocity_s': velocity_s,
            'pcc': pcc,
            'pvalue': pvalue,
            'inference_time_mean': np.mean(combined_infer_time),
            'inference_time_std': np.std(combined_infer_time),
            'sample_num': np.size(aligned_gt, 0)
        }

        # Store the metrics for each single sample
        all_metrics_single[suffix] = {
            'pmpjpe': pmpjpe,
            'angle': stat_angle_error,
            'velocity': velocity,
            'pcc': pcc_single,
        }

        # Initialize dictionary for keypoint-specific metrics
        keypoints_metrics[suffix] = {}
        keypoints_metrics_single[suffix] = {}

        # Calculate and store metrics for each keypoint
        num_keypoints = aligned_gt.shape[2]
        for i in range(num_keypoints):
            gt_keypoint = aligned_gt[:, :, i]
            pred_keypoint = aligned_pred[:, :, i]
            gt_keypoint = gt_keypoint[:, :, np.newaxis]
            pred_keypoint = pred_keypoint[:, :, np.newaxis]

            # Calculate metrics for this keypoint
            pmpjpe_m_kp, pmpjpe_s_kp = metrics.calculate_mpjpe(gt_keypoint, pred_keypoint)
            velocity_m_kp, velocity_s_kp = metrics.mean_velocity_error(gt_keypoint, pred_keypoint, sample_rate=25)
            pcc_kp, pvalue_kp = calculate_joint_correlations(gt_keypoint, pred_keypoint)

            # Calculate metrics in parallel using multiprocessing
        #    pmpjpe, velocity, pcc_single = calculate_metrics_for_sample(gt_keypoint, pred_keypoint, sample_rate=25)

        #    keypoints_metrics_single[suffix][gt_names[0][i]] = {
        #        'pmpjpe': pmpjpe,
        #        'velocity': velocity,
        #        'pcc': pcc_single,
        #    }

            # Store keypoint-specific metrics
            keypoints_metrics[suffix][gt_names[0][i]] = {
                'pmpjpe_m': pmpjpe_m_kp,
                'pmpjpe_s': pmpjpe_s_kp,
                'velocity_m': velocity_m_kp,
                'velocity_s': velocity_s_kp,
                'pcc': pcc_kp,
                'pvalue': pvalue_kp,
                'sample_num': np.size(gt_keypoint, 1)
            }
    return all_metrics, keypoints_metrics, all_metrics_single, keypoints_metrics_single


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
    all_metrics, keypoints_metrics, all_metrics_single, keypoints_metrics_single = (
        process_and_calculate_metrics_for_groups(grouped_files, directory))

    # Compare metrics of each group with 'none.pkl' group
    p_values = compare_metrics_with_none_group(all_metrics_single)

    # Convert results to DataFrames and save to CSV
    all_metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
    all_metrics_df.to_csv('all_metrics.csv')

    keypoints_metrics_df = pd.DataFrame.from_dict({(i, j): keypoints_metrics[i][j]
                                                   for i in keypoints_metrics.keys()
                                                   for j in keypoints_metrics[i].keys()}, orient='index')
    keypoints_metrics_df.to_csv('keypoints_metrics.csv')

    all_metrics_single_df = pd.DataFrame.from_dict(all_metrics_single, orient='index')
    all_metrics_single_df.to_csv('all_metrics_single.csv')

    keypoints_metrics_single_df = pd.DataFrame.from_dict({(i, j): keypoints_metrics_single[i][j]
                                                          for i in keypoints_metrics_single.keys()
                                                          for j in keypoints_metrics_single[i].keys()}, orient='index')
    keypoints_metrics_single_df.to_csv('keypoints_metrics_single.csv')

    p_values_df = pd.DataFrame.from_dict({(i, j): p_values[i][j]
                                          for i in p_values.keys()
                                          for j in range(len(p_values[i]))}, orient='index')
    p_values_df.to_csv('p_values.csv')

    # Print the p-values to see if any differences are statistically significant
    for suffix, pvals in p_values.items():
        print(f"Statistical significance of differences for group {suffix}:")
        print(pvals)

    # Now, metrics_results contains metrics for each group, you can log them to wandb or print them
    for suffix, metrics in all_metrics.items():
        print(f"Overall metrics for group {suffix}:")
        print(metrics)

    for suffix, metrics in keypoints_metrics.items():
        print(f"Keypoint-specific metrics for group {suffix}:")
        print(metrics)


main()
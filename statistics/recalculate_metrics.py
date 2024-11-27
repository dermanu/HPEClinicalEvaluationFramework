import os
import numpy as np
import pickle
from scipy import stats
import statsmodels.stats.api as sms
import multiprocessing
import pandas as pd
from utils import angle_metrics_tpose
from utils import metrics
import pingouin as pg
from statsmodels.stats.multitest import multipletests

# 0_4_5 fl, fm, fr x
# 5_4_1_3 bl, sl, fl, fm x
# 5_4_1 sl, fl, fm x
# 5_1 fm, sl x
# 4_3 bl, fl x
# 4_1_3
# 4_0 fl, fr  x
# 0_4_5


def is_data_normal(data, alpha=0.05):
    if len(data) > 5000:
        stat, p_value = stats.kstest(data, 'norm')
    else:
        stat, p_value = stats.shapiro(data)

    normal = p_value > alpha

    # Dataset large enough, thus CLT applies
    normal = True;
    return normal


def count_nan_frames(pred_frames):
    """
    Count the number of frames containing NaN values for all joints in both the ground truth and predicted arrays.
    :param pred_frames: Ground truth keypoints array [frames, joints, 3]
    :return: Number of frames with NaN values for any joint
    """
    # Check for NaN values in both the ground truth and predicted arrays
    nan_pred_frames = np.all(np.isnan(pred_frames), axis=(0, 2))

    # Count the total number of frames with NaN values
    nan_frame_count = np.sum(nan_pred_frames)
    return nan_frame_count


def bootstrap_confidence_interval(data1, data2, n_resamples=1000, alpha=0.05, paired=True):
    """
    Calculate the bootstrap confidence interval for the mean difference.
    """
    boot_diffs = []
    np.random.seed(0)  # For reproducibility
    for _ in range(n_resamples):
        if paired:
            indices = np.random.choice(len(data1), len(data1), replace=True)
            resample1 = data1[indices]
            resample2 = data2[indices]
            diff = np.mean(resample1 - resample2)
        else:
            resample1 = np.random.choice(data1, size=len(data1), replace=True)
            resample2 = np.random.choice(data2, size=len(data2), replace=True)
            diff = np.mean(resample1) - np.mean(resample2)
        boot_diffs.append(diff)
    lower = np.percentile(boot_diffs, (alpha/2)*100)
    upper = np.percentile(boot_diffs, (1 - alpha/2)*100)
    return lower, upper

def effect_size_interpreter(effect_size, effect_size_type):
    if effect_size_type == 'Cohen_d':
        if np.abs(effect_size) < 0.2:
            effect_interpret = 'negligible'
        elif 0.2 <= np.abs(effect_size) < 0.5:
            effect_interpret = 'small'
        elif 0.5 <= np.abs(effect_size) < 0.8:
            effect_interpret = 'medium'
        else:
            effect_interpret = 'large'
    elif effect_size_type == 'RBC':  # Rank-Biserial Correlation
        if np.abs(effect_size) < 0.1:
            effect_interpret = 'negligible'
        elif 0.1 <= np.abs(effect_size) < 0.3:
            effect_interpret = 'small'
        elif 0.3 <= np.abs(effect_size) < 0.5:
            effect_interpret = 'medium'
        else:
            effect_interpret = 'large'
    elif effect_size_type == 'Rosenthal':
        if np.abs(effect_size) < 0.1:
            effect_interpret = 'negligible'
        elif 0.1 <= np.abs(effect_size) < 0.3:
            effect_interpret = 'small'
        elif 0.3 <= np.abs(effect_size) < 0.5:
            effect_interpret = 'medium'
        else:
            effect_interpret = 'large'
    elif effect_size_type == 'Cliff':
        if np.abs(effect_size) < 0.147:
            effect_interpret = 'negligible'
        elif 0.147 <= np.abs(effect_size) < 0.33:
            effect_interpret = 'small'
        elif 0.33 <= np.abs(effect_size) < 0.474:
            effect_interpret = 'medium'
        else:
            effect_interpret = 'large'
    else:
        effect_interpret = np.nan

    return effect_interpret

def apply_bh_correction(p_values, alpha=0.05):
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    corrected_p_values = np.zeros_like(p_values)
    bh_significant = np.zeros_like(p_values, dtype=bool)

    for i, index in enumerate(sorted_indices):
        corrected_alpha = (i + 1) / m * alpha
        corrected_p_values[index] = p_values[index] <= corrected_alpha
        bh_significant[index] = p_values[index] <= corrected_alpha

    return bh_significant, corrected_p_values

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
    pool = multiprocessing.Pool(processes=20)

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


def compare_metrics_with_none_group(data, default_camera='none', alpha=0.05):
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
    baseline = data[default_camera]
    other_conditions = {k: v for k, v in data.items() if k != default_camera}
    metrics_data = baseline.keys()
    results = {}
    unpaired_conditions = ['lower', 'upper', 'complex', 'sitting']

    for metric in metrics_data:
        results[metric] = []
        p_values = []
        test_results = []
        base_data = np.array(baseline[metric]).flatten()

        normal_base = is_data_normal(base_data)

        for condition_name, condition_data in other_conditions.items():
            aug_data = np.array(condition_data[metric]).flatten()

            normal_aug = is_data_normal(aug_data)

            base_mean = np.mean(base_data)
            aug_mean = np.mean(aug_data)
            diff_percent = ((base_mean - aug_mean) / base_mean) * 100 if base_mean != 0 else 0
            diff_total = base_mean - aug_mean

            # Bootstrap CI for mean difference
            conf_int = bootstrap_confidence_interval(base_data, aug_data, n_resamples=1000, alpha=alpha,
                                                     paired=False)

            if condition_name in unpaired_conditions:
                # Unpaired data
                if normal_base and normal_aug:
                    t_result = pg.ttest(base_data, aug_data, paired=False, alternative='two-sided', correction='auto',
                                        r=0.5, confidence=0.95)
                    p_value = t_result['p-val'].iloc[0]
                    effect_size = t_result['cohen-d'].iloc[0]
                    if isinstance(t_result['CI95%'], pd.Series):
                        ci_lower = t_result['CI95%'].iloc[0][0]
                        ci_upper = t_result['CI95%'].iloc[0][1]
                    else:
                        ci_lower = t_result['CI95%'][0][0]
                        ci_upper = t_result['CI95%'][0][1]
                    effect_interpret = effect_size_interpreter(effect_size, 'Cohen_d')
                    test_used = 'Independent Samples t-test'
                else:
                    # Mann-Whitney U test with Scipy
                    mwu_result = pg.mwu(base_data, aug_data, alternative='two-sided')
                    p_value = mwu_result['p-val'].iloc[0]
                    effect_size = mwu_result['RBC'].iloc[0]
                    ci_lower, ci_upper = None, None
                    effect_interpret = effect_size_interpreter(effect_size,
                                                               'RBC')
                    test_used = 'Mann-Whitney U'

                p_values.append(p_value)

                # Unpaired metrics calculations
                # ccc = metrics.calculate_ccc(base_data, aug_data)

                # Significant flag will be updated after correction
                test_results.append({
                    'augmentation': condition_name,
                    'test_used': test_used,
                    'Significant': None,
                    'p_value': p_value,
                    'confidence_interval': conf_int,
                    'effect_size': effect_size,
                    'effect_size_ci': (ci_lower, ci_upper),
                    'effect_interpretation': effect_interpret,
                    'bias_percent': diff_percent,
                    'bias': diff_total,
                    # Paired-only metrics set to None
                    'pearsons_r': None,
                    'icc': None,
                    'icclb': None,
                    'iccup': None,
                    'sem': None,
                    'mdc': None
                })

            else:
                # Paired data
                if len(base_data) != len(aug_data):
                    print(
                        f"Warning: Length mismatch for {metric} - {condition_name}. Expected paired data but lengths differ.")
                    continue

                if normal_base and normal_aug and is_data_normal(base_data - aug_data):
                    t_result = pg.ttest(base_data, aug_data, paired=True, alternative='two-sided', correction='auto',
                                        r=0.5, confidence=0.95)
                    p_value = t_result['p-val'].iloc[0]
                    effect_size = t_result['cohen-d'].iloc[0]
                    if isinstance(t_result['CI95%'], pd.Series):
                        ci_lower = t_result['CI95%'].iloc[0][0]
                        ci_upper = t_result['CI95%'].iloc[0][1]
                    else:
                        ci_lower = t_result['CI95%'][0][0]
                        ci_upper = t_result['CI95%'][0][1]
                    effect_interpret = effect_size_interpreter(effect_size, 'Cohen_d')
                    test_used = 'Paired t-test'
                else:
                    t_result = pg.wilcoxon(base_data, aug_data, alternative='two-sided')
                    p_value = t_result['p-val']
                    effect_size = t_result['RBC']
                    ci_lower, ci_upper = None, None
                    test_used = 'Wilcoxon signed-rank test'

                p_values.append(p_value)

                # Paired metrics calculations
                icc, icclb, iccup = metrics.calculate_icc(base_data, aug_data)
                sem, mdc = metrics.calculate_sem_mdc(np.std(aug_data - base_data), icc)
                pearson_r, _ = metrics.calculate_pearson_r(base_data, aug_data)

                # Significant flag will be updated after correction
                test_results.append({
                    'augmentation': condition_name,
                    'test_used': test_used,
                    'Significant': None,
                    'p_value': p_value,
                    'confidence_interval': conf_int,
                    'effect_size': effect_size,
                    'effect_size_ci': (ci_lower, ci_upper),
                    'effect_interpretation': effect_interpret,
                    'bias_percent': diff_percent,
                    'bias': diff_total,
                    'pearsons_r': pearson_r,
                    'icc': icc,
                    'icclb': icclb,
                    'iccup': iccup,
                    'sem': sem,
                    'mdc': mdc
                })

        # Apply multiple comparison correction
        bh_significant, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

        # Update significance and corrected p-values
        for i, idx in enumerate(np.argsort(p_values)):
            test_results[idx]['Significant'] = bh_significant[i]
            test_results[idx]['corrected_p_value'] = corrected_p_values[i]
            # You can interpret effect size here if needed
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


def align_procrustes_per_frame(gt, pred):
    """
    Procrustes alignment per frame to avoid large memory allocations.
    :param gt: Ground truth 3D joint positions for a single frame [joint, 3]
    :param pred: Predicted 3D joint positions for a single frame [joint, 3]
    :return: Aligned predicted joint positions for a single frame [joint, 3]
    """
    if np.sum(np.abs(pred)) != 0:
        try:
            muX = np.mean(pred, axis=0)
            muY = np.mean(gt, axis=0)

            X0 = pred - muX
            Y0 = gt - muY

            var1 = np.sum(X0 ** 2)
            K = X0.T.dot(Y0)  # Much smaller matrix, size (3, 3)
            U, s, Vh = np.linalg.svd(K)
            V = Vh.T
            Z = np.eye(U.shape[0])
            Z[-1, -1] *= np.sign(np.linalg.det(U.dot(Vh)))
            R = V.dot(Z.dot(U.T))
            scale = np.trace(R.dot(K)) / var1
            t = muY - scale * (R.dot(muX))
            pred_hat = scale * R.dot(pred.T).T + t
        except np.linalg.LinAlgError:
            pred_hat = np.full(gt.shape, np.nan)  # Handle failed alignment
    else:
        pred_hat = np.full(gt.shape, np.nan)  # Handle empty predictions

    return pred_hat


def align_procrustes_old(target, prediction):
    """
    Procrustes MPJPE: MPJPE after rigid alignment for each frame individually.
    :param target: Ground truth 3D joint positions, shape [frames, joint, 3]
    :param prediction: Predicted 3D joint positions, shape [frames, joint, 3]
    :return: Aligned ground truth and prediction, shape [frames, joint, 3], error count of failed alignments
    """
    gt_all, pred_all = [], []
    error_count = 0

    for gt_frame, pred_frame in zip(target, prediction):
        pred_aligned = align_procrustes_per_frame(gt_frame, pred_frame)
        if np.any(np.isnan(pred_aligned)):
            error_count += 1
        gt_all.append(gt_frame)
        pred_all.append(pred_aligned)

    gt_all = np.array(gt_all)
    pred_all = np.array(pred_all)

    if error_count > 0:
        print(f"Procrustes alignment failed {error_count} times")

    return gt_all, pred_all, error_count


def group_files_by_suffix(files):
    """
    Group files by the augmentation (modifier), e.g., 'background', 'defocus', etc.
    :param files: List of filenames.
    :return: Dictionary where keys are augmentations (modifiers) and values are lists of files.
    """
    from collections import defaultdict
    grouped_files = defaultdict(list)

    for file in files:
        # Remove the '.pkl' extension and split the filename
        parts = file.replace('.pkl', '').split('-')

        # Extract the modifier (augmentation)
        modifier = '-'.join(parts[2:])
        grouped_files[modifier].append(file)

        # Extract the modifier (task)
        if modifier == 'none':
            modifier = '-'.join(parts[1:2])
            grouped_files[modifier].append(file)


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
    angle_errors_metrics = {}

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

        # Count number of NANs
        nan_frame_count = count_nan_frames(combined_pred_keypoints)

        # Align keypoints using Procrustes alignment
        aligned_gt, aligned_pred, error_count = align_procrustes_old(combined_gt_keypoints, combined_pred_keypoints)

        # Swap axes to match with the expected input from metrics
        aligned_gt = np.moveaxis(aligned_gt, [0, 1, 2], [2, 0, 1])
        aligned_pred = np.moveaxis(aligned_pred, [0, 1, 2], [2, 0, 1])

        #print('GT_length proc')
        #print(np.size(aligned_gt, 0))
        #print('Pred_length proc')
        #print(np.size(aligned_pred, 0))

        # Calculate metrics
        angles_gt = angle_metrics_tpose.calculate_angles_tpose(aligned_gt)
        angles_pred = angle_metrics_tpose.calculate_angles_tpose(aligned_pred)
        angle_errors = {
            joint: np.mean(np.abs(np.degrees(np.abs(angles_gt[joint] - angles_pred[joint]))), axis=-1) for joint in
            angles_gt.keys()
        }
        all_angle_error_m = { joint: np.mean(angle_errors[joint], axis=0)
                             for joint in angle_errors.keys() }
        all_angle_error_s = { joint: np.std(angle_errors[joint], axis=0)
                             for joint in angle_errors.keys() }

        angle_errors_metrics[suffix] = {
            'all_angle_error_m': all_angle_error_m,
            'all_angle_error_s': all_angle_error_s
        }

        all_angle_errors = np.concatenate([angle_errors[joint].flatten() for joint in angle_errors.keys()])
        angle_m = np.mean(all_angle_errors)
        angle_s = np.std(all_angle_errors)

        # Combine the results across all joints by averaging across the frames for all joints for the statistics
        stat_angle_error = np.std(list(angle_errors.values()), axis=0)

        pmpjpe_m, pmpjpe_s = metrics.calculate_mpjpe(aligned_gt, aligned_pred)
        velocity_m, velocity_s = metrics.mean_velocity_error(aligned_gt, aligned_pred, sample_rate=25)
        pcc, pvalue = calculate_joint_correlations(aligned_gt, aligned_pred)

        #print('GT_length angle')
        #print(np.size(angles_gt['right_shoulder_angles'], 0))
        #print('Pred_length angle')
        #print(np.size(angles_pred['right_shoulder_angles'], 0))

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
            'inference_time_mean': np.mean(combined_infer_time),
            'inference_time_std': np.std(combined_infer_time),
            'sample_num': np.size(aligned_gt, 0),
            'sample_nan_num': nan_frame_count
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

            # Store keypoint-specific metrics
            keypoints_metrics[suffix][gt_names[0][i]] = {
                'pmpjpe_m': pmpjpe_m_kp,
                'pmpjpe_s': pmpjpe_s_kp,
                'velocity_m': velocity_m_kp,
                'velocity_s': velocity_s_kp,
                'pcc': pcc_kp,
                'sample_num': np.size(gt_keypoint, 1)
            }

    return all_metrics, keypoints_metrics, all_metrics_single, keypoints_metrics_single, angle_errors_metrics


def main():
    """
    Run inference on the chosen model with sweep parameters and log results to wandb project.
    """
    directory = '/home/emanu/Desktop/multi/combined'

    # List of all files in the directory
    all_files = os.listdir(directory)

    # Group files by suffix
    grouped_files = group_files_by_suffix(all_files)

    # Process each group and calculate metrics
    all_metrics, keypoints_metrics, all_metrics_single, keypoints_metrics_single, angle_errors_metrics = (
        process_and_calculate_metrics_for_groups(grouped_files, directory)
    )

    with open('angle_errors_metrics.pkl', 'wb') as f:
        pickle.dump(angle_errors_metrics, f)

    # Saving the variables as .pkl files
    with open('all_metrics.pkl', 'wb') as f:
        pickle.dump(all_metrics, f)

    with open('keypoints_metrics.pkl', 'wb') as f:
        pickle.dump(keypoints_metrics, f)

    with open('all_metrics_single.pkl', 'wb') as f:
        pickle.dump(all_metrics_single, f)

    with open('keypoints_metrics_single.pkl', 'wb') as f:
        pickle.dump(keypoints_metrics_single, f)

    # Compare metrics of each group with baseline group
    perform_statistical_analysis(all_metrics_single)

    p_values = compare_metrics_with_none_group(all_metrics_single, default_camera='cameras_4_0')

    with open('p_values.pkl', 'wb') as f:
        pickle.dump(p_values, f)

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

    data_for_df = []
    for suffix, angle_data in angle_errors_metrics.items():
        for angle_name in angle_data['all_angle_error_m'].keys():
            data_for_df.append({
                'suffix': suffix,
                'angle_name': angle_name,
                'angle_error_mean': angle_data['all_angle_error_m'][angle_name],
                'angle_error_std': angle_data['all_angle_error_s'][angle_name]
            })

    # Create DataFrame
    angle_errors_df = pd.DataFrame(data_for_df)

    # Save to CSV
    angle_errors_df.to_csv('angle_errors_metrics.csv', index=False)


    # Print the p-values to see if any differences are statistically significant
    for suffix, pvals in p_values.items():
        print(f"Statistical significance of differences for group {suffix}:")
        print(pvals)

    # Plot metrics_results contains metrics for each group
    for suffix, metrics_data in all_metrics.items():
        print(f"Overall metrics for group {suffix}:")
        print(metrics_data)

    for suffix, metrics_data in keypoints_metrics.items():
        print(f"Keypoint-specific metrics for group {suffix}:")
        print(metrics_data)




def main_load():
    # Load metrics
    with open('single/all_metrics_single.pkl', 'rb') as f:
        all_metrics_single = pickle.load(f)

    with open('single/all_metrics.pkl', 'rb') as f:
        all_metrics = pickle.load(f)

    with open('single/keypoints_metrics.pkl', 'rb') as f:
        keypoints_metrics = pickle.load(f)

    with open('single/keypoints_metrics_single.pkl', 'rb') as f:
        keypoints_metrics_single = pickle.load(f)


    # Compare metrics of each group with 'none.pkl' group

    perform_statistical_analysis(all_metrics_single)
    #p_values = compare_metrics_with_none_group(all_metrics_single, default_camera='cameras_5')

    #with open('p_values.pkl', 'wb') as f:
    #    pickle.dump(p_values, f)

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

    # Plot metrics_results contains metrics for each group
    for suffix, metrics_data in all_metrics.items():
        print(f"Overall metrics for group {suffix}:")
        print(metrics_data)

    for suffix, metrics_data in keypoints_metrics.items():
        print(f"Keypoint-specific metrics for group {suffix}:")
        print(metrics_data)

def perform_statistical_analysis(all_metrics_single, metric_name='angle'):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import levene
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    import pingouin as pg
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    import numpy as np

    # Prepare the DataFrame
    metrics_list = []
    movement_types = ['upper', 'lower', 'complex', 'sitting']

    for movement_type in movement_types:
        if movement_type in all_metrics_single:
            metrics = all_metrics_single[movement_type]
            metric_values = metrics[metric_name]  # Use the metric_name variable
            for value in metric_values:
                metrics_list.append({
                    'movement_type': movement_type,
                    'metric_value': value[0] if isinstance(value, (list, np.ndarray)) else value  # Adjust indexing if necessary
                })
        else:
            print(f"Warning: Movement type '{movement_type}' not found in all_metrics_single.")

    # Create DataFrame
    df = pd.DataFrame(metrics_list)

    # Save the DataFrame for reporting
    df.to_csv(f'{metric_name}_movement_type_metrics.csv', index=False)

    # Check homogeneity of variances
    grouped_data = [group['metric_value'].values for _, group in df.groupby('movement_type')]
    stat, p = levene(*grouped_data)
    print(f"Levene's test p-value: {p}")

    # Save Levene's test results
    levene_results_df = pd.DataFrame({
        'statistic': [stat],
        'p_value': [p]
    })
    levene_results_df.to_csv(f'{metric_name}_levene_test_results.csv', index=False)

    if p < 0.05:
        print("Variances are unequal across groups. Proceeding with Welch's ANOVA.")
        # Perform Welch's ANOVA
        welch_anova_results = pg.welch_anova(dv='metric_value', between='movement_type', data=df)
        print(welch_anova_results)
        # Save Welch's ANOVA results
        welch_anova_results.to_csv(f'{metric_name}_anova_results.csv', index=False)
        # Extract Partial Eta Squared
        eta_squared = welch_anova_results['np2'][0]
        print(f"Partial Eta Squared: {eta_squared:.5f}")
        interpretation = interpret_eta_squared(eta_squared)
        # Save effect size with interpretation
        effect_size_df = pd.DataFrame({
            'Effect_Size_Type': ['Partial Eta Squared'],
            'Effect_Size_Value': [eta_squared],
            'Interpretation': [interpretation]
        })
        effect_size_df.to_csv(f'{metric_name}_effect_size.csv', index=False)
        # Post-hoc test: Games-Howell
        posthoc_results = pg.pairwise_gameshowell(dv='metric_value', between='movement_type', data=df)
        print(posthoc_results)
        # Save post-hoc test results
        posthoc_results.to_csv(f'{metric_name}_posthoc_results.csv', index=False)
    else:
        print("Variances are equal across groups. Proceeding with One-Way ANOVA.")
        # Fit the model
        model = ols('metric_value ~ C(movement_type)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)
        # Save ANOVA table
        anova_table.to_csv(f'{metric_name}_anova_results.csv', index=True)
        # Calculate effect size (eta squared)
        eta_squared = anova_table['sum_sq']['C(movement_type)'] / anova_table['sum_sq'].sum()
        print(f"Eta Squared: {eta_squared:.5f}")
        interpretation = interpret_eta_squared(eta_squared)
        # Save effect size with interpretation
        effect_size_df = pd.DataFrame({
            'Effect_Size_Type': ['Eta Squared'],
            'Effect_Size_Value': [eta_squared],
            'Interpretation': [interpretation]
        })
        effect_size_df.to_csv(f'{metric_name}_effect_size.csv', index=False)
        # Post-hoc test: Tukey's HSD
        tukey = pairwise_tukeyhsd(endog=df['metric_value'], groups=df['movement_type'], alpha=0.05)
        print(tukey)
        # Convert tukey results to DataFrame for better display
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        print(tukey_df)
        # Save Tukey HSD results
        tukey_df.to_csv(f'{metric_name}_posthoc_results.csv', index=False)

    # Visualize the results
    sns.boxplot(x='movement_type', y='metric_value', data=df)
    plt.title(f'Measurement Accuracy Across Movement Types ({metric_name.capitalize()})')
    plt.xlabel('Movement Type')
    plt.ylabel(metric_name.capitalize())
    plt.tight_layout()
    plt.savefig(f'{metric_name}_boxplot.png')
    plt.show()

def interpret_eta_squared(eta_squared):
    if eta_squared < 0.01:
        interpretation = 'small effect'
    elif 0.01 <= eta_squared < 0.06:
        interpretation = 'medium effect'
    elif 0.06 <= eta_squared < 0.14:
        interpretation = 'large effect'
    else:
        interpretation = 'very large effect'
    print(f"Effect Size Interpretation: {interpretation}")
    return interpretation


def interpret_eta_squared(eta_squared):
    if eta_squared < 0.01:
        interpretation = 'small effect'
    elif 0.01 <= eta_squared < 0.06:
        interpretation = 'medium effect'
    elif 0.06 <= eta_squared < 0.14:
        interpretation = 'large effect'
    else:
        interpretation = 'very large effect'
    print(f"Effect Size Interpretation: {interpretation}")


main()


import os
import sys
import pickle

from keras.src.utils.module_utils import scipy
from scipy import stats
import multiprocessing
from statsmodels.stats.multitest import multipletests
import pandas as pd
import seaborn as sns
from scipy.stats import levene
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, rankdata
import scikit_posthocs as sp


sys.path.append('./')
from utils import angle_metrics_tpose
from utils import metrics


def is_data_normal(data, alpha=0.05, plot=True, filename='example.png'):
    """ Test if the data follows a normal distribution using the Shapiro-Wilk or
    Kolmogorov-Smirnov test."""
    if len(data) > 5000:
        stat, p_value = stats.kstest(data, 'norm')
    else:
        stat, p_value = stats.shapiro(data)

    # Large samples often considered normal by CLT
    normal = (p_value > alpha)

    if plot:
        ax = pg.qqplot(data, dist='norm')
        plt.title("Q–Q Plot")
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        normal = False # Just for now

    return normal

def count_nan_frames(pred_frames):
    """ Count the number of frames containing NaN values.

    Parameters:
    - pred_frames: Array of predicted frames [frames, joints, 3].

    Returns:
    - Number of frames with NaN values.
    """
    nan_pred_frames = np.all(np.isnan(pred_frames), axis=(0, 2))
    nan_frame_count = np.sum(nan_pred_frames)
    return nan_frame_count

def bootstrap_confidence_interval(data1, data2, n_resamples=1000, alpha=0.05, paired=True):
    """
    Calculate a bootstrap confidence interval for the mean difference between two datasets.

    Parameters:
    - data1, data2: Arrays of data to compare.
    - n_resamples: Number of bootstrap resamples.
    - alpha: Significance level.
    - paired: Boolean indicating if the data is paired.

    Returns:
    - lower, upper: Confidence interval bounds.
    """
    boot_diffs = []
    np.random.seed(0)

    for _ in range(n_resamples):
        if paired:
            indices = np.random.choice(len(data1), len(data1), replace=True)
            resample1, resample2 = data1[indices], data2[indices]
            diff = np.mean(resample1 - resample2)
        else:
            resample1 = np.random.choice(data1, size=len(data1), replace=True)
            resample2 = np.random.choice(data2, size=len(data2), replace=True)
            diff = np.mean(resample1) - np.mean(resample2)
        boot_diffs.append(diff)

    lower = np.percentile(boot_diffs, (alpha / 2) * 100)
    upper = np.percentile(boot_diffs, (1 - alpha / 2) * 100)
    return lower, upper

def effect_size_interpreter(effect_size, effect_size_type):
    """
    Interpret the magnitude of an effect size based on its type.

    Parameters:
    - effect_size: Numeric effect size value.
    - effect_size_type: Type of effect size (e.g., 'Cohen_d', 'RBC').

    Returns:
    - Effect size interpretation as a string.
    """
    thresholds = {
        'Cohen_d': [(0.2, 'small'), (0.5, 'medium'), (0.8, 'large')],
        'RBC': [(0.1, 'small'), (0.3, 'medium'), (0.5, 'large')],
        'Rosenthal': [(0.1, 'small'), (0.3, 'medium'), (0.5, 'large')],
        'Cliff': [(0.147, 'small'), (0.33, 'medium'), (0.474, 'large')],
        'epsilon_squared': [(0.01, 'small'), (0.06, 'medium'), (0.14, 'large')]
    }

    for threshold, label in thresholds.get(effect_size_type, []):
        if np.abs(effect_size) < threshold:
            return label
    return np.nan

def apply_bh_correction(p_values, alpha=0.05):
    """
    Apply the Benjamini-Hochberg correction to a list of p-values.

    Parameters:
    - p_values: List of p-values.
    - alpha: Significance level.

    Returns:
    - reject: Boolean array indicating rejected null hypotheses.
    - p_values_corrected: Adjusted p-values. Currently adjusted to all and not within groups
                            (movement, camera placement, augmentation)
    """

    # m = len(p_values)
    # sorted_indices = np.argsort(p_values)
    # corrected_p_values = np.zeros_like(p_values)
    # bh_significant = np.zeros_like(p_values, dtype=bool)
    #
    # for i, index in enumerate(sorted_indices):
    #     corrected_alpha = (i + 1) / m * alpha
    #     corrected_p_values[index] = p_values[index] <= corrected_alpha
    #     bh_significant[index] = p_values[index] <= corrected_alpha

    reject, p_values_corrected = pg.multicomp(p_values, alpha=alpha, method='fdr_bh')
    return reject, p_values_corrected

def align_procrustes_per_frame(gt, pred):
    """
    Perform Procrustes alignment on predicted keypoints relative to ground truth for a single frame.

    Parameters:
    - gt: Ground truth keypoints for one frame [joints, 3].
    - pred: Predicted keypoints for one frame [joints, 3].

    Returns:
    - Aligned predicted keypoints.
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

    Parameters:
   - target: Ground truth 3D joint positions, shape [frames, joint, 3]
   - prediction: Predicted 3D joint positions, shape [frames, joint, 3]

    Returns:
    - gt_all: Ground truth, shape [frames, joint, 3]
    - pred_all: Aligned prediction, shape [frames, joint, 3]
    - error_count: Error count of failed alignments
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

def sliding_window(arr, window_size, step_size):
    """
    Generates sliding windows over the temporal axis of a 3D array.

    Parameters:
    - arr: 3D array with shape [time, joint, 3]
    - window_size: Size of the sliding window
    - step_size: Step size between windows

    Returns:
    - Generator yielding windows of the original array
    """
    for start in range(0, arr.shape[0] - window_size + 1, step_size):
        yield arr[start:start + window_size]

def worker_mpjpe(inputs):
    """
    Allows for the parallelization of the MPJPE calculation

    Parameters:
    - inputs: Array of both aligned ground truth and predicted keypoints.

    Returns:
    - MPJPE for each frame/sample
    """

    aligned_gt_sample, aligned_pred_sample = inputs

    # Compute MPJPE (calculate_mpjpe expects shape [num_keypoints, 3] with axis=1)
    mean, _, _ = metrics.calculate_mpjpe(aligned_gt_sample, aligned_pred_sample, axis=0)
    return mean


def worker_velocity_and_corr(inputs):
    """
    Allows for the parallelization of the velocity and correlation calculation. Not really used
    anymore. Legacy.

    Parameters:
    - inputs: Array of both aligned ground truth and predicted keypoints.

    Returns:
    - Velocity for each window
    - PCC for each window
    """

    gt_window, pred_window, sample_rate = inputs
    velocity = metrics.mean_velocity_error(gt_window, pred_window, sample_rate=sample_rate)
    corr, _ = calculate_joint_correlations(gt_window, pred_window) # PCC
    return velocity, corr

def calculate_metrics_for_sample(aligned_gt, aligned_pred, sample_rate, window_size=26, step_size=13):
    """
    Calculate spatiotemporal metrics for a single sample using windowing using parallelization. Not really used anymore.
    Legacy.

    Parameters:
    - aligned_gt: Procrustes-aligned ground truth keypoints.
    - aligned_pred: Procrustes-aligned prediction keypoints
    - sample_rate: Sample rate of dataset
    - window_size: Size of the sliding window
    - step_size: Step size between windows

    Returns:
    - pmpjpe: PMPJPE for each frame/sample
    - velocity: Velocity for each window
    - pcc: PCC for each window
    """

    # Setup multiprocessing pool
    pool = multiprocessing.Pool(processes=15)

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

def hodges_lehmann_estimator_and_ci(diff, base_reference=None, alpha=0.05, n_samples=10**5):
    """
    Approximate the Hodges–Lehmann estimator and its confidence interval using Monte Carlo sampling,
    and compute the bias as a percentage relative to a baseline reference.

    Parameters:
      - diff: array of paired differences (e.g. base_data - aug_data).
      - base_reference: baseline value (e.g., median of base_data) to express bias as percentage.
                        If None, percentage bias is not computed.
      - alpha: significance level (default=0.05).
      - n_samples: Number of random Walsh averages to compute.

    Returns:
      - hl_est: the HL estimator (median of all Walsh averages).
      - (ci_lower, ci_upper): confidence interval for the HL estimator (percentile-based).
      - (hl_est_percent, ci_lower_percent, ci_upper_percent): percentage bias estimates, or None if base_reference is None.
    """
    diff = np.asarray(diff)
    n = len(diff)
    total_pairs = n * (n + 1) // 2

    # If total pairs are fewer than n_samples, compute exactly.
    if total_pairs <= n_samples:
        walsh = []
        for i in range(n):
            for j in range(i, n):
                walsh.append((diff[i] + diff[j]) / 2.0)
        walsh = np.array(walsh)
    else:
        # Monte Carlo approximation of Walsh averages.
        walsh = np.empty(n_samples)
        for k in range(n_samples):
            i = np.random.randint(0, n)
            j = np.random.randint(i, n)
            walsh[k] = (diff[i] + diff[j]) / 2.0

    hl_est = np.median(walsh)
    ci_lower = np.percentile(walsh, 100 * (alpha / 2))
    ci_upper = np.percentile(walsh, 100 * (1 - alpha / 2))

    if base_reference is not None and base_reference != 0:
        hl_est_percent = (hl_est / base_reference) * 100
    else:
        hl_est_percent = None

    return hl_est, (ci_lower, ci_upper), hl_est_percent

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
    # Prepare inputs
    baseline = data[default_camera]
    other_conditions = {k: v for k, v in data.items() if k != default_camera}
    metrics_data = baseline.keys()
    results = {}
    unpaired_conditions = ['lower', 'upper', 'complex', 'sitting']

    # Run statistical analysis for every metric
    for metric in metrics_data:
        results[metric] = []
        p_values = []
        test_results = []
        base_data = np.array(baseline[metric]).flatten()

        # Check for normal distribution
        normal_base = is_data_normal(base_data, plot=True, filename='baseline_'+metric+'.png')

        # Run statistical analysis for every augmentation/camera placement
        for condition_name, condition_data in other_conditions.items():
            aug_data = np.array(condition_data[metric]).flatten()

            # Check for normal distribution
            normal_aug = is_data_normal(aug_data, plot=True, filename=condition_name+'_'+metric+'.png')

            # Calculate means
            base_mean = np.nanmean(base_data)
            aug_mean = np.nanmean(aug_data)

            # Calculate total and proportional bias.
            diff_percent = ((base_mean - aug_mean) / base_mean) * 100 if base_mean != 0 else 0
            diff_total = base_mean - aug_mean

            # Bootstrap CI for mean difference
            conf_int = bootstrap_confidence_interval(base_data, aug_data, n_resamples=1000, alpha=alpha,
                                                     paired=False)

            # Run statistical analysis specifically for unpaired conditions
            if condition_name in unpaired_conditions:
                if normal_base and normal_aug:
                    # Run unpaired two-sided t-test (p-value, cohen's d, lower and upper ci bounds)
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
                    # Interpret effect size
                    effect_interpret = effect_size_interpreter(effect_size, 'Cohen_d')
                    # Save which test was used
                    test_used = 'Independent Samples t-test'
                else:
                    # Two-sided Mann-Whitney-U test (p-value, RBC)
                    U, p_value = stats.mannwhitneyu(base_data, aug_data, alternative='two-sided', method='asymptotic')
                    n1 = len(base_data)
                    n2 = len(aug_data)
                    # Compute Rank Biserial Correlation (RBC) as effect size
                    effect_size = 1 - (2 * U) / (n1 * n2)
                    ci_lower, ci_upper = None, None
                    # mwu_result = pg.mwu(base_data, aug_data, alternative='two-sided')
                    # p_value = mwu_result['p-val'].iloc[0]
                    # effect_size = mwu_result['RBC'].iloc[0]
                    # ci_lower, ci_upper = None, None
                    # Interpret effect size
                    effect_interpret = effect_size_interpreter(effect_size,
                                                               'RBC')
                    # Save which test was used
                    test_used = 'Mann-Whitney U'

                p_values.append(p_value)

                # Save metrics in a dict
                test_results.append({
                    'augmentation': condition_name,
                    'test_used': test_used,
                    'Significant': None,
                    'p_value': p_value,
                    'confidence_interval': conf_int,
                    'effect_size': effect_size,
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
                # Run statistical analysis specifically for paired conditions
                # A little sanity check that at least the sample number is equal
                if len(base_data) != len(aug_data):
                    print(
                        f"Warning: Length mismatch for {metric} - {condition_name}. Expected paired data but lengths differ.")
                    continue

                # Check if data is normal distributed:
                if normal_base and normal_aug:
                    # Run paired two-sided t-test (p-value, cohen's d, lower and upper ci bounds)
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
                    # Run Wilcoxon-test (p-value, RBC)
                    d = base_data - aug_data
                    stat, p_value = wilcoxon(d, alternative='two-sided', method='approx') # Can't be calculated exactly. To large dataset.
                    # Compute Rank Biserial Correlation (RBC)
                    nonzero = d != 0
                    d_nz = d[nonzero]
                    abs_d = abs(d_nz)
                    # Assign ranks to the absolute differences
                    ranks = rankdata(abs_d)
                    # Sum ranks corresponding to positive differences
                    W = ranks[d_nz > 0].sum()
                    # Total rank sum for nonzero differences; for n observations, T = n(n+1)/2
                    T = ranks.sum()
                    effect_size = (2 * W - T) / T
                    # diff_total, (ci_lower, ci_upper), diff_percent = hodges_lehmann_estimator_and_ci(d, base_median,  alpha=alpha)
                    test_used = 'Wilcoxon signed-rank test'
                    effect_interpret = effect_size_interpreter(effect_size, 'RBC')

                    # t_result = pg.wilcoxon(base_data, aug_data, alternative='two-sided')
                    # p_value = t_result['p-val'].iloc[0]
                    # effect_size = t_result['RBC'].iloc[0]
                    # hl_est, (ci_lower, ci_upper) = hodges_lehmann_estimator_and_ci(base_data-aug_data, alpha=alpha)
                    #
                    # test_used = 'Wilcoxon signed-rank test'
                    # effect_interpret = effect_size_interpreter(effect_size, 'RBC')
                p_values.append(p_value)

                # Additional paired metrics (ICC, SEM, MED, Pearson R). Not used yet.
                icc, icclb, iccup = metrics.calculate_icc(base_data, aug_data)
                sem, mdc = metrics.calculate_sem_mdc(np.std(aug_data - base_data), icc)
                pearson_r, _ = metrics.calculate_pearson_r(base_data, aug_data)

                # Save metrics in a dict
                test_results.append({
                    'augmentation': condition_name,
                    'test_used': test_used,
                    'Significant': None,
                    'p_value': p_value,
                    'confidence_interval': conf_int,
                    'effect_size': effect_size,
                    'effect_interpretation': effect_interpret,
                    'bias_percent': diff_percent,
                    'bias': diff_total,
                    'pearsons_r': pearson_r,
                    'icc': icc,
                    'icclb': icclb,
                    'iccup': iccup,
                    'sem': sem,
                    'mdc': mdc,
                })

        # Apply multiple comparison correction. Currently accros all groups. Should be only within groups.
        # (camera placements, setup errors)
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
    Helper-function to convert a dictionary of joint keypoints to a numpy array.

    Parameters:
    - joint_dict: Dictionary where keys are joint names and values are 3D coordinates.

    Returns:
    - Numpy array with shape [joints, 3].
    - Keys of joint_dict
    """
    return np.array(list(joint_dict.values())), list(joint_dict.keys())

def calculate_joint_correlations(target_array, pred_array):
    """
        Calculate the average correlation and p-value between ground truth and predicted joint positions
        across all keypoints in a 3D pose estimation setup.

        Parameters:
        - target_array (numpy.ndarray): A 3D array of ground truth joint positions with shape
          [samples, XYZ, keypoints].
        - pred_array (numpy.ndarray): A 3D array of predicted joint positions with the same shape as target_array.

        Returns:
        - mean_correlation (float): The average correlation coefficient across all keypoints.
        - mean_pvalue (float): The average p-value for the correlation test across all keypoints.
    """

    joint_correlations = []
    joint_pvalues = []
    num_keypoints = target_array.shape[2]

    # Calculate correlation and p-value for each joint
    for joint in range(num_keypoints):
        target = target_array[:, :, joint]
        prediction = pred_array[:, :, joint]

        # Calculate correlation using the provided function
        correlation, pvalue = metrics.calculate_correlation(target, prediction)
        joint_correlations.append(correlation)
        joint_pvalues.append(pvalue)

    return np.mean(joint_correlations), np.mean(joint_pvalues)

def group_files_by_suffix(files):
    """
    Group files by the augmentation, e.g., 'background', 'defocus', etc.

    Parameters:
    - files: List of filenames.

    Returns:
    - Dictionary where keys are augmentations (modifiers) and values are lists of files.
    """
    from collections import defaultdict
    grouped_files = defaultdict(list)

    for file in files:
        parts = file.replace('.pkl', '').split('-')
        modifier = '-'.join(parts[2:])
        grouped_files[modifier].append(file)

        if modifier == 'none':
            modifier = '-'.join(parts[1:2])
            grouped_files[modifier].append(file)

    return grouped_files

def process_and_calculate_metrics_for_groups(grouped_files, directory):
    """
    Process each group of files, calculate metrics, and return raw keypoints.

    Parameters:
    - grouped_files: Dictionary of grouped files by suffix.
    - directory: Directory containing the files.

    Returns:
     - Dictionary with calculated metrics, keypoint-specific metrics, and raw keypoints for each group.
    """
    # Initialize dictionaries
    all_metrics = {}
    all_metrics_single = {}
    keypoints_metrics = {}
    keypoints_metrics_single = {}
    angle_errors_metrics = {}

    # Calculate metrics for all augmentations (camera placements, setup errors, movement types)
    for suffix, files in grouped_files.items():
        print(f"Calculating metrics for conditions with suffix: {suffix}")
        combined_pred_keypoints = []
        combined_gt_keypoints = []
        combined_infer_time = []

        # Read and combine data from all files in the respective group
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

        # Count number of NANs (not used yet)
        nan_frame_count = count_nan_frames(combined_pred_keypoints)

        # Align keypoints using Procrustes alignment
        aligned_gt, aligned_pred, error_count = align_procrustes_old(combined_gt_keypoints, combined_pred_keypoints)

        # Swap axes to match with the expected input from metrics
        aligned_gt = np.moveaxis(aligned_gt, [0, 1, 2], [2, 0, 1])
        aligned_pred = np.moveaxis(aligned_pred, [0, 1, 2], [2, 0, 1])

        # Calculate metrics
        angles_gt = angle_metrics_tpose.calculate_angles_tpose(aligned_gt)
        angles_pred = angle_metrics_tpose.calculate_angles_tpose(aligned_pred)
        angle_errors = {
            joint: np.mean(np.abs(np.degrees(np.abs(angles_gt[joint] - angles_pred[joint]))), axis=-1) for joint in
            angles_gt.keys()
        }

        # Calculate mean and std acros all joint angles (overall MPJAE)
        all_angle_error_m = { joint: np.mean(angle_errors[joint], axis=0)
                             for joint in angle_errors.keys() }
        all_angle_error_s = { joint: np.std(angle_errors[joint], axis=0)
                             for joint in angle_errors.keys() }

        angle_errors_metrics[suffix] = {
            'all_angle_error_m': all_angle_error_m,
            'all_angle_error_s': all_angle_error_s
        }

        all_angle_errors = np.concatenate([angle_errors[joint].flatten() for joint in angle_errors.keys()])

        # Calculate MPJAVE
        angle_velocity_error = angle_metrics_tpose.calculate_angular_speed_error(angles_gt, angles_pred, 25, post_smoothing=True)

        # Calculate statistics for plotting boxplot later
        angle_m = np.mean(all_angle_errors)
        angle_s = np.std(all_angle_errors)
        Q1, median, Q3 = np.percentile(all_angle_errors, [25, 50, 75])
        IQR = Q3 - Q1
        loval = (Q1 - 1.5 * IQR)
        hival = (Q3 + 1.5 * IQR)
        wiskhi = np.compress(all_angle_errors <= hival, all_angle_errors)
        wisklo = np.compress(all_angle_errors >= loval, all_angle_errors)
        actual_hival = np.max(wiskhi)
        actual_loval = np.min(wisklo)
        angle_Qs = [Q1, median, Q3, loval, hival, actual_loval, actual_hival]

        # Combine the results across all joints by averaging across the frames for all joints for the statistics
        stat_angle_error = np.mean(list(angle_errors.values()), axis=0)

        # Calculate MPJPE and PCC
        pmpjpe_m, pmpjpe_s, pmpjpe_Qs = metrics.calculate_mpjpe(aligned_gt, aligned_pred)
        pcc, pvalue = calculate_joint_correlations(aligned_gt, aligned_pred)

        # Calculate metrics in parallel using multiprocessing
        pmpjpe, velocity, pcc_single = calculate_metrics_for_sample(aligned_gt, aligned_pred, sample_rate=25)

        # Store the metrics in dict
        all_metrics[suffix] = {
            'pmpjpe_m': pmpjpe_m,
            'pmpjpe_s': pmpjpe_s,
            'pmpjpe_Qs': pmpjpe_Qs,
            'angle_m': angle_m,
            'angle_s': angle_s,
            'angle_Qs': angle_Qs,
            'velocity_m': angle_velocity_error['mean'],
            'velocity_s': angle_velocity_error['std'],
            'velocity_Qs': [angle_velocity_error['Q1'], angle_velocity_error['median'], angle_velocity_error['Q3'], angle_velocity_error['loval'], angle_velocity_error['hival'], angle_velocity_error['actual_loval'], angle_velocity_error['actual_hival']],
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
            'velocity': list(map(lambda x: x[0], velocity)), # NOT ANGLE VELOCITY ERROR YET!
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
            pmpjpe_m_kp, pmpjpe_s_kp, Qs = metrics.calculate_mpjpe(gt_keypoint, pred_keypoint)

            pcc_kp, pvalue_kp = calculate_joint_correlations(gt_keypoint, pred_keypoint)

            # Store keypoint-specific metrics
            keypoints_metrics[suffix][gt_names[0][i]] = {
                'pmpjpe_m': pmpjpe_m_kp,
                'pmpjpe_s': pmpjpe_s_kp,
                'pcc': pcc_kp,
                'sample_num': np.size(gt_keypoint, 1)
            }

    return all_metrics, keypoints_metrics, all_metrics_single, keypoints_metrics_single, angle_errors_metrics

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


def perform_statistical_analysis(all_metrics_single, metric_name='angle', directory="./"):
    """
    Perform statistical analysis for the given metric across different movement types.

    Parameters:
    - all_metrics_single (dict): Dictionary containing metrics data for each movement type.
    - metric_name (str): The name of the metric to analyze (default is 'angle').
    """
    # Prepare the data for analysis
    metrics_list = []
    movement_types = ['upper', 'lower', 'complex', 'sitting']

    # Extract metric values for each movement type
    for movement_type in movement_types:
        if movement_type in all_metrics_single:
            metrics = all_metrics_single[movement_type]
            metric_values = metrics[metric_name]
            for value in metric_values:
                metrics_list.append({
                    'movement_type': movement_type,
                    'metric_value': value[0] if isinstance(value, (list, np.ndarray)) else value
                })
        else:
            print(f"Warning: Movement type '{movement_type}' not found in all_metrics_single.")

    # Create a DataFrame for statistical analysis and save it
    df = pd.DataFrame(metrics_list)
    df.to_csv(os.path.join(directory, f'{metric_name}_movement_type_metrics.csv'), index=False)

    if False: # Should test for normal distribution in the future
        # Perform the Kruskal–Wallis test
        kruskal_results = pg.kruskal(data=df, dv='metric_value', between='movement_type')
        # Extract the test statistic and uncorrected p-value
        H = kruskal_results['H'][0]
        p_kw = kruskal_results['p-unc'][0]

        # Save Kruskal-Wallis results
        kw_results_df = pd.DataFrame({'H_statistic': [H], 'p_value': [p_kw]})
        kw_results_df.to_csv(os.path.join(directory, f'{metric_name}_kruskal_results.csv'), index=False)

        # Compute epsilon-squared effect size: (H - (k-1))/(N - k)
        N = len(df)
        k = df['movement_type'].nunique()
        epsilon_squared = (H - (k - 1)) / (N - k) if (N - k) != 0 else np.nan
        print(f"Epsilon Squared: {epsilon_squared:.5f}")
        interpretation = interpret_eta_squared(epsilon_squared)  # Adjust thresholds if needed.
        effect_size_df = pd.DataFrame({
            'Effect_Size_Type': ['Epsilon Squared'],
            'Effect_Size_Value': [epsilon_squared],
            'Interpretation': [interpretation]
        })
        effect_size_df.to_csv(os.path.join(directory, f'{metric_name}_effect_size.csv'), index=False)

        # Conduct pairwise post-hoc comparisons using Dunn's test
        posthoc_results = sp.posthoc_dunn(df, val_col='metric_value', group_col='movement_type', p_adjust='fdr_bh')
        print(posthoc_results)
        posthoc_results.to_csv(os.path.join(directory, f'{metric_name}_posthoc_results.csv'))
    else:
        # Test for homogeneity of variances using Levene's test
        grouped_data = [group['metric_value'].values for _, group in df.groupby('movement_type')]
        stat_levene, p_levene = levene(*grouped_data)
        print(f"Levene's test p-value: {p_levene}")
        levene_results_df = pd.DataFrame({'statistic': [stat_levene], 'p_value': [p_levene]})
        levene_results_df.to_csv(os.path.join(directory, f'{metric_name}_levene_test_results.csv'), index=False)

        # Determine the appropriate ANOVA test based on the homogeneity of variances
        if p_levene < 0.05:
            print("Variances are unequal across groups. Proceeding with Welch's ANOVA.")
            # Perform Welch's ANOVA
            welch_anova_results = pg.welch_anova(dv='metric_value', between='movement_type', data=df)
            print(welch_anova_results)
            # Save Welch's ANOVA results
            welch_anova_results.to_csv(os.path.join(directory, f'{metric_name}_anova_results.csv'), index=False)
            # Extract Partial Eta Squared and interpret its value
            eta_squared = welch_anova_results['np2'][0]
            print(f"Partial Eta Squared: {eta_squared:.5f}")
            interpretation = interpret_eta_squared(eta_squared)
            # Save effect size and its interpretation
            effect_size_df = pd.DataFrame({
                'Effect_Size_Type': ['Partial Eta Squared'],
                'Effect_Size_Value': [eta_squared],
                'Interpretation': [interpretation]
            })
            effect_size_df.to_csv(os.path.join(directory, f'{metric_name}_effect_size.csv'), index=False)
            # Conduct Games-Howell post-hoc test
            posthoc_results = pg.pairwise_gameshowell(dv='metric_value', between='movement_type', data=df)
            print(posthoc_results)
            # Save post-hoc test results
            posthoc_results.to_csv(os.path.join(directory, f'{metric_name}_posthoc_results.csv'), index=False)
        else:
            print("Variances are equal across groups. Proceeding with One-Way ANOVA.")
            # Fit the One-Way ANOVA model
            model = ols('metric_value ~ C(movement_type)', data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print(anova_table)
            # Save ANOVA results
            anova_table.to_csv(os.path.join(directory, f'{metric_name}_anova_results.csv'), index=True)
            # Calculate Eta Squared and interpret its value
            eta_squared = anova_table['sum_sq']['C(movement_type)'] / anova_table['sum_sq'].sum()
            print(f"Eta Squared: {eta_squared:.5f}")
            interpretation = interpret_eta_squared(eta_squared)
            # Save effect size and its interpretation
            effect_size_df = pd.DataFrame({
                'Effect_Size_Type': ['Eta Squared'],
                'Effect_Size_Value': [eta_squared],
                'Interpretation': [interpretation]
            })
            effect_size_df.to_csv(os.path.join(directory, f'{metric_name}_effect_size.csv'), index=False)
            # Conduct Tukey's HSD post-hoc test
            tukey = pairwise_tukeyhsd(endog=df['metric_value'], groups=df['movement_type'], alpha=0.05)
            print(tukey)
            # Save Tukey's HSD results
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            print(tukey_df)
            tukey_df.to_csv(os.path.join(directory, f'{metric_name}_posthoc_results.csv'), index=False)


def main():
    """
    Command-line interface for metrics analysis.
    """
    parser = argparse.ArgumentParser(description="Run metrics analysis for human pose estimation.")
    parser.add_argument("directory", type=str, help="Directory containing the data files.")
    parser.add_argument("output", type=str, help="Directory to save output results.")
    args = parser.parse_args()

    np.random.seed(42)

    # List all files in the specified directory
    all_files = os.listdir(args.directory)

    # Group files by their suffix
    grouped_files = group_files_by_suffix(all_files)

    # # Process each group and calculate metrics
    # all_metrics, keypoints_metrics, all_metrics_single, keypoints_metrics_single, angle_errors_metrics\
    #     = process_and_calculate_metrics_for_groups(grouped_files, args.directory)
    #
    # # Save results to the output directory
    # os.makedirs(args.output, exist_ok=True)
    # with open(os.path.join(args.output,'angle_errors_metrics.pkl'), 'wb') as f:
    #     pickle.dump(angle_errors_metrics, f)
    #
    # with open(os.path.join(args.output,'all_metrics.pkl'), 'wb') as f:
    #     pickle.dump(all_metrics, f)
    #
    # with open(os.path.join(args.output,'keypoints_metrics.pkl'), 'wb') as f:
    #     pickle.dump(keypoints_metrics, f)
    #
    # with open(os.path.join(args.output, 'all_metrics_single.pkl'), 'wb') as f:
    #     pickle.dump(all_metrics_single, f)

    with open(os.path.join(args.output,'all_metrics_single.pkl'), 'rb') as f:
        all_metrics_single = pickle.load(f)

    # Compare metrics of each group with baseline group
    print("Statistics on movement categories")
    perform_statistical_analysis(all_metrics_single)

    print("Statistics on augmentation and camera placement")
    p_values = compare_metrics_with_none_group(all_metrics_single, default_camera='none')

    p_values_df = pd.DataFrame.from_dict({(i, j): p_values[i][j]
                                          for i in p_values.keys()
                                          for j in range(len(p_values[i]))}, orient='index')
    p_values_df.to_csv(os.path.join(args.output, 'p_values.csv'), index=True)

    print(f"Analysis complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()

# Example command
# python statistics/recalculate_metrics.py /home/user/Desktop/multi/combined /statistics/multi_new

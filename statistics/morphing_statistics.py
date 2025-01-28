import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, levene
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats
import argparse

def calculate_pa_mpjpe(ground_truths, predictions):
    """Calculate PA-MPJPE mean, standard deviation, and joint errors."""
    n_samples = ground_truths.shape[0]
    n_joints = ground_truths.shape[1] // 3  # Assuming 3D keypoints
    ground_truths = ground_truths.reshape(n_samples, n_joints, 3)
    predictions = predictions.reshape(n_samples, n_joints, 3)
    joint_errors = np.linalg.norm(ground_truths - predictions, axis=2)
    return joint_errors.mean(axis=1), joint_errors.std(axis=1), joint_errors

def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = group1.mean(), group2.mean()
    std1, std2 = group1.std(ddof=1), group2.std(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_std

def load_data(ground_truth_path, hpe_path, prediction_path):
    """Load ground truths, HPE estimations, and predictions."""
    ground_truths = np.load(ground_truth_path)
    hpe_estimations = np.load(hpe_path)
    predictions = np.load(prediction_path)
    return ground_truths, hpe_estimations, predictions

def analyze_results(ground_truths, hpe_estimations, predictions, joint_names):
    """
    Perform analysis including PA-MPJPE, Cohen's d, and statistical tests to evaluate the morphing models performance.

    Parameters:
    ground_truths (str): Path to the ground truths .npy file of none augmented data.
    hpe_estimations (str): Path to the HPE estimations .npy file of none augmented data.
    predictions (str): Path to the morphed HPE estimations .npy file of none augmented data.
    joint_names (str): Array of joint names.

    Returns:
    None
    """

    # Calculate sample size
    sample_size = ground_truths.shape[0]

    # Calculate PA-MPJPE for Original and Morphed
    orig_mean, orig_std, orig_joint_errors = calculate_pa_mpjpe(ground_truths, hpe_estimations)
    morph_mean, morph_std, morph_joint_errors = calculate_pa_mpjpe(ground_truths, predictions)

    # Improvement in mean and std
    mean_improvement = ((orig_mean.mean() - morph_mean.mean()) / orig_mean.mean()) * 100
    std_improvement = ((orig_std.mean() - morph_std.mean()) / orig_std.mean()) * 100

    print(f"Sample size: {sample_size}")
    print(f"Original Mean PA-MPJPE: {orig_mean.mean():.2f} mm")
    print(f"Morphed Mean PA-MPJPE: {morph_mean.mean():.2f} mm")
    print(f"Improvement in Mean PA-MPJPE: {mean_improvement:.2f}%")
    print(f"Original Std PA-MPJPE: {orig_std.mean():.2f} mm")
    print(f"Morphed Std PA-MPJPE: {morph_std.mean():.2f} mm")
    print(f"Improvement in Std PA-MPJPE: {std_improvement:.2f}%")

    # Effect sizes (Cohen's d)
    joint_effect_sizes = []
    for i, joint_name in enumerate(joint_names):
        orig_errors = orig_joint_errors[:, i]
        morph_errors = morph_joint_errors[:, i]
        d = calculate_cohens_d(orig_errors, morph_errors)
        joint_effect_sizes.append({"Joint": joint_name, "Cohen's d": d})

    effect_size_df = pd.DataFrame(joint_effect_sizes)
    print("\nEffect Sizes (Cohen's d) for Joints:")
    print(effect_size_df)

    # Levene's Test for Equality of Variances
    levene_stat, levene_p = levene(orig_mean, morph_mean)
    equal_var = levene_p > 0.05
    print(f"\nLevene's Test: Statistic = {levene_stat:.2f}, p = {levene_p:.3f}")
    print("Variances are equal." if equal_var else "Variances are unequal.")

    # Independent Samples t-test
    t_stat, p_value = ttest_ind(orig_mean, morph_mean, equal_var=equal_var)
    print(f"\nT-Test for Mean PA-MPJPE Reduction:")
    print(f"T-Statistic = {t_stat:.2f}, P-Value = {p_value:.3f}")

    # Confidence Interval for Difference in Means
    se = np.sqrt(((orig_std.mean()**2) / sample_size) + ((morph_std.mean()**2) / sample_size))
    df = sample_size - 1
    t_critical = stats.t.ppf(1 - 0.025, df)
    margin_of_error = t_critical * se
    mean_diff = morph_mean.mean() - orig_mean.mean()
    ci_lower, ci_upper = mean_diff - margin_of_error, mean_diff + margin_of_error
    print(f"95% Confidence Interval for Difference in Means: ({ci_lower:.2f}, {ci_upper:.2f}) mm")

    # Pairwise significance for joints
    joint_significance = []
    p_values = []
    for i, joint_name in enumerate(joint_names):
        t_stat_joint, p_val_joint = ttest_ind(orig_joint_errors[:, i], morph_joint_errors[:, i], equal_var=False)
        joint_significance.append({"Joint": joint_name, "T-Statistic": t_stat_joint, "P-Value": p_val_joint})
        p_values.append(p_val_joint)

    # Apply Bonferroni correction
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
    significance_df = pd.DataFrame(joint_significance)
    significance_df['P-Value (Bonferroni Corrected)'] = pvals_corrected
    significance_df['Significant'] = reject

    print("\nJoint-wise Significance of MPJPE Reduction (Bonferroni Corrected):")
    print(significance_df)

# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="Analyze MPJPE and related statistics.")
    parser.add_argument("ground_truths_path", type=str, help="Path to the ground truths .npy file.")
    parser.add_argument("hpe_path", type=str, help="Path to the HPE estimations .npy file.")
    parser.add_argument("predictions_path", type=str, help="Path to the predictions .npy file.")
    args = parser.parse_args()

    # Define joint names
    joint_names = [
        'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow', 'right_wrist', 'left_wrist',
        'right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_ankle', 'left_ankle',
        'right_heel', 'left_heel', 'right_foot_index', 'left_foot_index'
    ]

    # Load data
    ground_truths, hpe_estimations, predictions = load_data(args.ground_truths_path, args.hpe_path, args.predictions_path)

    # Analyze results
    analyze_results(ground_truths, hpe_estimations, predictions, joint_names)

if __name__ == "__main__":
    main()

# Example command
# python statistics/morphing_statistics.py /home/emanu/Desktop/all_ground_truths.npy /home/emanu/Desktop/all_hpe_truths.npy /home/emanu/Desktop/all_predictions.npy

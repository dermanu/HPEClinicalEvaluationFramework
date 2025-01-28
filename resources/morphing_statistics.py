import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, levene
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load Data (replace with your file paths)
ground_truths = np.load('/home/emanu/Desktop/all_ground_truths.npy')
hpe_estimations = np.load('/home/emanu/Desktop/all_hpe_truths.npy')
predictions = np.load('/home/emanu/Desktop/all_predictions.npy')

# Define joint names
joint_names = [
    'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow', 'right_wrist', 'left_wrist',
    'right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_ankle', 'left_ankle',
    'right_heel', 'left_heel', 'right_foot_index', 'left_foot_index'
]

# Calculate the sample size
sample_size = ground_truths.shape[0]
print(f"Sample size: {sample_size}")

# Function to calculate PA-MPJPE
def calculate_pa_mpjpe(ground_truths, predictions):
    n_samples = ground_truths.shape[0]
    n_joints = ground_truths.shape[1] // 3  # Assuming 3D keypoints
    ground_truths = ground_truths.reshape(n_samples, n_joints, 3)
    predictions = predictions.reshape(n_samples, n_joints, 3)

    # Calculate joint errors
    joint_errors = np.linalg.norm(ground_truths - predictions, axis=2)
    return joint_errors.mean(axis=1), joint_errors.std(axis=1), joint_errors

# Calculate PA-MPJPE mean and std for Original and Morphed
orig_mean, orig_std, orig_joint_errors = calculate_pa_mpjpe(ground_truths, hpe_estimations)
morph_mean, morph_std, morph_joint_errors = calculate_pa_mpjpe(ground_truths, predictions)

# Improvement in mean and std
mean_improvement = ((orig_mean.mean() - morph_mean.mean()) / orig_mean.mean()) * 100
std_improvement = ((orig_std.mean() - morph_std.mean()) / orig_std.mean()) * 100

print(f"Original Mean PA-MPJPE: {orig_mean.mean():.2f} mm")
print(f"Morphed Mean PA-MPJPE: {morph_mean.mean():.2f} mm")
print(f"Improvement in Mean PA-MPJPE: {mean_improvement:.2f}%")
print(f"Original Std PA-MPJPE: {orig_std.mean():.2f} mm")
print(f"Morphed Std PA-MPJPE: {morph_std.mean():.2f} mm")
print(f"Improvement in Std PA-MPJPE: {std_improvement:.2f}%")

# Effect size calculation (Cohen's d) for each joint
def calculate_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = group1.mean(), group2.mean()
    std1, std2 = group1.std(ddof=1), group2.std(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_std

joint_effect_sizes = []
for i, joint_name in enumerate(joint_names):
    orig_errors = orig_joint_errors[:, i]
    morph_errors = morph_joint_errors[:, i]
    d = calculate_cohens_d(orig_errors, morph_errors)
    joint_effect_sizes.append({"Joint": joint_name, "Cohen's d": d})

effect_size_df = pd.DataFrame(joint_effect_sizes)
print("\nEffect Sizes (Cohen's d) for Joints:")
print(effect_size_df)

# Statistical significance of error reduction using Independent Samples t-test
# First, check for equality of variances using Levene's Test
levene_stat, levene_p = levene(orig_mean, morph_mean)
print(f"\nLevene's Test for Equality of Variances: Statistic = {levene_stat:.2f}, p = {levene_p:.3f}")

# Decide whether to assume equal variances
equal_var = True if levene_p > 0.05 else False
if equal_var:
    print("Variances are equal. Proceeding with standard t-test (equal_var=True).")
else:
    print("Variances are unequal. Proceeding with Welch's t-test (equal_var=False).")

# Perform Independent Samples t-test
t_stat, p_value = ttest_ind(orig_mean, morph_mean, equal_var=equal_var)
degrees_of_freedom = (len(orig_mean) + len(morph_mean) - 2) if equal_var else 'Welch-Satterthwaite'

print(f"\nIndependent Samples t-test for Mean PA-MPJPE Reduction:")
print(f"T-Statistic = {t_stat:.2f}")
print(f"P-Value = {p_value:.3f}")
print(f"Degrees of Freedom = {degrees_of_freedom}")

# Optional: Calculate Confidence Interval for the difference in means
import scipy.stats as stats

if equal_var:
    se = np.sqrt(((orig_std.mean() ** 2) / sample_size) + ((morph_std.mean() ** 2) / sample_size))
else:
    # Welch's t-test degrees of freedom
    se = np.sqrt((orig_std.mean() ** 2) / sample_size + (morph_std.mean() ** 2) / sample_size)
    # Approximate degrees of freedom using Welch-Satterthwaite equation
    df_num = ( (orig_std.mean() ** 2) / sample_size + (morph_std.mean() ** 2) / sample_size ) ** 2
    df_den = ( ((orig_std.mean() ** 2) / sample_size) ** 2 ) / (sample_size - 1) + \
             ( ((morph_std.mean() ** 2) / sample_size) ** 2 ) / (sample_size - 1)
    df = df_num / df_den
    degrees_of_freedom = df

confidence_level = 0.95
alpha = 1 - confidence_level
if equal_var:
    df = len(orig_mean) + len(morph_mean) - 2
else:
    df = df  # already calculated above

t_critical = stats.t.ppf(1 - alpha/2, df) if isinstance(df, (int, float)) else None
if t_critical is not None:
    margin_of_error = t_critical * se
    mean_diff = morph_mean.mean() - orig_mean.mean()
    ci_lower = mean_diff - margin_of_error
    ci_upper = mean_diff + margin_of_error
    print(f"95% Confidence Interval for the difference in means: ({ci_lower:.2f}, {ci_upper:.2f}) mm")
else:
    print("Could not calculate Confidence Interval due to undefined degrees of freedom.")

# Pairwise significance for joints (t-tests)
from statsmodels.stats.multitest import multipletests

joint_significance = []
p_values = []
for i, joint_name in enumerate(joint_names):
    t_stat_joint, p_val_joint = ttest_ind(orig_joint_errors[:, i], morph_joint_errors[:, i], equal_var=False)
    joint_significance.append({"Joint": joint_name, "T-Statistic": t_stat_joint, "P-Value": p_val_joint})
    p_values.append(p_val_joint)

# Convert to DataFrame
significance_df = pd.DataFrame(joint_significance)

# Apply Bonferroni correction for multiple comparisons
reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
significance_df['P-Value (Bonferroni Corrected)'] = pvals_corrected
significance_df['Significant'] = reject

print("\nJoint-wise Significance of MPJPE Reduction (Bonferroni Corrected):")
print(significance_df)

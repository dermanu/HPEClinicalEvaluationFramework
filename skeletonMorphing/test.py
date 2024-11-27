from scipy.spatial import procrustes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


joint_names = [
    'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow', 'right_wrist', 'left_wrist', 'right_hip',
    'left_hip', 'right_knee', 'left_knee', 'right_ankle', 'left_ankle', 'right_heel', 'left_heel', 'right_foot_index',
    'left_foot_index'
]

def calculate_pa_mpjpe(ground_truths, hpe_estimations, predictions):
    """
    Calculate the Procrustes Aligned Mean Per Joint Position Error (PA-MPJPE).

    Args:
        ground_truths (np.ndarray): Ground truth keypoints [n_samples, n_joints * 3].
        hpe_estimations (np.ndarray): Estimated keypoints [n_samples, n_joints * 3].
        predictions (np.ndarray): Predicted keypoints [n_samples, n_joints * 3].

    Returns:
        float: PA-MPJPE.
        dict: Joint-wise MPJPE.
    """
    n_samples = ground_truths.shape[0]
    joint_dim = ground_truths.shape[1] // 3  # Assuming 3D keypoints

    # Reshape to [n_samples, n_joints, 3]
    ground_truths_reshaped = ground_truths.reshape(n_samples, joint_dim, 3)
    hpe_estimations_reshaped = hpe_estimations.reshape(n_samples, joint_dim, 3)
    predictions_reshaped = predictions.reshape(n_samples, joint_dim, 3)

    pa_mpjpe = 0.0
    joint_errors = {joint_name: 0.0 for joint_name in joint_names}

    for i in range(n_samples):
        # Align the estimated keypoints to the ground truth using Procrustes analysis
        #_, aligned_estimation, _ = procrustes(ground_truths_reshaped[i], hpe_estimations_reshaped[i])

        # Compute MPJPE for this sample
        error = np.linalg.norm(ground_truths_reshaped[i] - hpe_estimations_reshaped[i], axis=1)
        pa_mpjpe += error.mean()

        # Compute joint-wise errors
        for idx, joint_name in enumerate(joint_names):
            joint_error_gt_to_hpe = np.linalg.norm(ground_truths_reshaped[i, idx] - hpe_estimations_reshaped[i, idx])

            joint_error_pred_to_gt = np.linalg.norm(predictions_reshaped[i, idx] - ground_truths_reshaped[i, idx])
            joint_errors[joint_name] += joint_error_gt_to_hpe - joint_error_pred_to_gt

    # Compute average PA-MPJPE
    pa_mpjpe /= n_samples
    joint_errors = {joint_name: joint_error / n_samples for joint_name, joint_error in joint_errors.items()}

    print(f"Samples: {n_samples}")

    return pa_mpjpe, joint_errors

# Load the provided .npy files
ground_truths_path = '/home/emanu/Desktop/all_ground_truths.nyp.npy'
hpe_estimations_path = '/home/emanu/Desktop/all_hpe_truths.nyp.npy'
prediction_path = '/home/emanu/Desktop/all_predictions.nyp.npy'

ground_truths = np.load(ground_truths_path)
hpe_estimations = np.load(hpe_estimations_path)
predictions = np.load(prediction_path)

# Calculate PA-MPJPE
pa_mpjpe_result, joint_errors_result = calculate_pa_mpjpe(ground_truths, hpe_estimations, predictions)

print(f"PA-MPJPE: {pa_mpjpe_result}")
print("Joint-wise Errors:")
for joint, error in joint_errors_result.items():
    print(f"{joint}: {error}")


# Prepare data for plotting
joint_names = list(joint_errors_result.keys())
joint_error_pred_to_gt = []
joint_error_hpe_to_gt = []

# Calculate joint errors for each joint across samples
n_samples = ground_truths.shape[0]
joint_dim = ground_truths.shape[1] // 3
ground_truths_reshaped = ground_truths.reshape(n_samples, joint_dim, 3)
hpe_estimations_reshaped = hpe_estimations.reshape(n_samples, joint_dim, 3)
predictions_reshaped = predictions.reshape(n_samples, joint_dim, 3)

for idx, joint_name in enumerate(joint_names):
    pred_to_gt_errors = [
        np.linalg.norm(predictions_reshaped[i, idx] - ground_truths_reshaped[i, idx])
        for i in range(n_samples)
    ]
    hpe_to_gt_errors = [
        np.linalg.norm(hpe_estimations_reshaped[i, idx] - ground_truths_reshaped[i, idx])
        for i in range(n_samples)
    ]
    joint_error_pred_to_gt.append(pred_to_gt_errors)
    joint_error_hpe_to_gt.append(hpe_to_gt_errors)

# Prepare data for paired boxplot
formatted_joint_names = [
    "R Shoulder", "L Shoulder", "R Elbow", "L Elbow", "R Wrist", "L Wrist",
    "R Hip", "L Hip", "R Knee", "L Knee", "R Ankle", "L Ankle",
    "R Heel", "L Heel", "R Foot Index", "L Foot Index"
]
paired_data = []
for joint_idx, joint_name in enumerate(joint_names):
    for i in range(n_samples):
        paired_data.append({"Joint": joint_name, "Type": "Original", "Error": joint_error_hpe_to_gt[joint_idx][i]})
        paired_data.append({"Joint": joint_name, "Type": "Morphed", "Error": joint_error_pred_to_gt[joint_idx][i]})

# Convert to DataFrame
paired_df = pd.DataFrame(paired_data)
joint_name_mapping = dict(zip(joint_names, formatted_joint_names))
paired_df["Joint"] = paired_df["Joint"].map(joint_name_mapping)

# Remove outliers based on IQR
def remove_outliers(df, column_name, group_column):
    def filter_group(group):
        q1 = group[column_name].quantile(0.25)
        q3 = group[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return group[(group[column_name] >= lower_bound) & (group[column_name] <= upper_bound)]

    return df.groupby(group_column, group_keys=False).apply(filter_group)


filtered_paired_df = remove_outliers(paired_df,
                                     "Error", "Joint")


# Define the new order for joints
new_joint_order = [
    "L Shoulder", "R Shoulder", "L Elbow", "R Elbow", "L Wrist", "R Wrist",
    "L Hip", "R Hip", "L Knee", "R Knee", "L Ankle", "R Ankle",
    "L Heel", "R Heel", "L Foot Index", "R Foot Index"
]

# Convert "Joint" column to a categorical type with the specified order
filtered_paired_df["Joint"] = pd.Categorical(
    filtered_paired_df["Joint"], categories=new_joint_order, ordered=True
)

filtered_paired_df.to_pickle('filtered_paired_df.pkl')


# Plot using seaborn for better grouping and color differentiation
plt.figure(figsize=(15, 8))
sns.violinplot(
    data=filtered_paired_df,
    x="Joint",
    y="Error",
    hue="Type",
    split=True,
    inner="quart",
    gap=.1,
    cut=0,
    palette={"Morphed": "blue", "Original": "red"},
)
#plt.title("MPJPE Distribution: Prediction to GT vs HPE to GT", fontsize=16)
plt.xlabel("")
plt.ylabel("MPJPE (Error)", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title="Type", title_fontsize=14, fontsize=12)
plt.tight_layout()
plt.show()
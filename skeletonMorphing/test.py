from scipy.spatial import procrustes
import numpy as np

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

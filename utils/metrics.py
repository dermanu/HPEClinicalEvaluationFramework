import numpy as np
from scipy import signal, stats
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def align_by_pelvis(joints):
    """
    Align the input joints by the pelvis joint
    :param joints: 3D joint positions
    :return: Aligned 3D joint positions
    """
    pelvis = joints[:, 0, :]
    joints = joints - pelvis[:, np.newaxis, :]
    return joints


def calculate_mpjpe(target, prediction):
    """
    Mean per-joint position error (MPJPE)
    :param target: Ground truth 3D joint positions
    :param prediction: Predicted 3D joint positions
    :return: Mean and standard deviation of the MPJPE
    """
    assert prediction.shape == target.shape, "The shape of prediction and target must match."

    mpjpe = np.linalg.norm(prediction - target, axis=2)
    mean = np.mean(mpjpe)
    std = np.std(mpjpe)

    return mean, std


def align_procrustes(target, prediction):
    """
    Procrustes MJPE: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    Based on the implementation from https://github.com/miraymen/3dpw-eval/blob/master/evaluate.py
    :param target: Ground truth 3D joint positions, shape [sample, joint, 3]
    :param prediction: Predicted 3D joint positions, shape [sample, joint, 3]
    :return gt_all: Ground truth 3D joint positions after alignment, shape [sample, joint, 3]
    :return pred_all: Predicted 3D joint positions after alignment, shape [sample, joint, 3]
    :return error_count: Error count of failed alignments
    """

    gt_all = []
    pred_all = []
    error_count = 0
    joint_number = target.shape[1]

    for (gt, pred) in (zip(target, prediction)):
        gt_raw = gt
        if not (np.sum(np.abs(pred)) == 0):
            transposed = False
            if pred.shape[0] != 3 and pred.shape[0] != 2:
                pred = pred.T
                gt = gt.T
                transposed = True
            assert (gt.shape[1] == pred.shape[1]), "The number of joints must match."

            # 1. Remove mean.
            muX = np.mean(pred, axis=1, keepdims=True)
            muY = np.mean(gt, axis=1, keepdims=True)
            X0 = pred - muX
            Y0 = gt - muY

            # 2. Compute variance of X1 used for scale.
            var1 = np.sum(X0 ** 2)

            # 3. The outer product of X1 and X2.
            K = X0.dot(Y0.T)

            # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
            try:
                U, s, Vh = np.linalg.svd(K)
            except np.linalg.LinAlgError:
                # print("SVD did not converge")
                error_count += 1
                continue

            V = Vh.T
            # Construct Z that fixes the orientation of R to get det(R)=1.
            Z = np.eye(U.shape[0])
            Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
            # Construct R.
            R = V.dot(Z.dot(U.T))

            # 5. Recover scale.
            scale = np.trace(R.dot(K)) / var1

            # 6. Recover translation.
            t = muY - scale * (R.dot(muX))

            # 7. Error:
            pred_hat = scale * R.dot(pred) + t

            if transposed:
                pred_hat = pred_hat.T

        else:
            pred_hat = np.tile(np.mean(gt, axis=0), (joint_number, 1))
            R = np.identity(3)

        gt_all.append(gt_raw)
        pred_all.append(pred_hat)

    gt_all = np.array(gt_all)
    pred_all = np.array(pred_all)

    if error_count > 0:
        print(f"Procrustes alignment failed {error_count} times")

    return gt_all, pred_all, error_count


def calculate_pmpjpe(target, prediction):
    """
    Procrustes MPJPE: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    :param target: Ground truth 3D joint positions
    :param prediction: Predicted 3D joint positions
    :return: Mean and standard deviation of the PMPJPE
    :return: Error count of failed alignments
    """
    assert prediction.shape == target.shape, "The shape of prediction and target must match."

    target, prediction, error_count = align_procrustes(target, prediction)
    mean, std = calculate_mpjpe(target, prediction)

    return mean, std, error_count


def calculate_pck(target, prediction, threshold=100.0,
                  joints_to_use=None, procrustes=False):
    """
    Calculate percentage of correct keypoints (PCK) in [%] (https://arxiv.org/pdf/1611.09813.pdf)
    :param target: Ground truth 3D joint positions
    :param prediction: Predicted 3D joint positions
    :param threshold: Threshold for correct keypoints
    :param joints_to_use: List of joints to use for PCK calculation
    :param procrustes: Whether to use procrustes alignment
    :return: PCK in [%]
    """

    if joints_to_use is None:
        joints_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert len(prediction.shape) == len(target.shape), "The shape of prediction and target must match."

    if procrustes:
        target, prediction, err = align_procrustes(target, prediction)

    distance = np.linalg.norm(target - prediction, axis=2)

    pck = distance <= threshold
    pck = np.mean(pck[:, joints_to_use], axis=1)
    pck = np.mean(pck) * 100.

    return pck


def mean_velocity_error(prediction, target, sample_rate, procrustes=True):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    :param prediction: Predicted 3D joint positions
    :param target: Ground truth 3D joint positions
    :param sample_rate: Sample rate in Hz
    :param procrustes: Weather Procrustes alignment should be used
    :param procrustes: Whether to use procrustes alignment
    :return: Mean and standard deviation of the mean per-joint velocity error
    """

    assert prediction.shape == target.shape, "The shape of prediction and target must match."

    velocity_predicted = np.diff(prediction, axis=0) * sample_rate
    velocity_target = np.diff(target, axis=0) * sample_rate

    if procrustes:
        mean, std, err = calculate_pmpjpe(velocity_target, velocity_predicted)
    else:
        mean, std, err = calculate_mpjpe(velocity_target, velocity_predicted)

    return mean, std


def mean_acceleration_error(prediction, target, sample_rate, procrustes=True):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    :param prediction: Predicted 3D joint positions
    :param target: Ground truth 3D joint positions
    :param procrustes: Whether to use procrustes alignment
    :param sample_rate: Sample rate in Hz
    :param procrustes: Weather Procrustes alignment should be used
    :return: Mean and standard deviation of the mean per-joint velocity error
    """

    assert prediction.shape == target.shape, "The shape of prediction and target must match."

    if procrustes:
        target, prediction, err = align_procrustes(target, prediction)

    acceleration_predicted = np.diff(np.diff(prediction, axis=0), axis=0) * sample_rate
    acceleration_target = np.diff(np.diff(target, axis=0), axis=0) * sample_rate

    mean, std, err = calculate_pmpjpe(acceleration_target, acceleration_predicted)

    return mean, std


def calculate_signal_similarity(target, prediction, axes_to_use=None, procrustes = False):
    """
    Calculate average Pearson correlation coefficient for all joints and axes to measure signal similarity.
    :param target: Ground truth 3D joint positions, shape [sample, joint, 3]
    :param prediction: Predicted 3D joint positions, shape [sample, joint, 3]
    :param axes_to_use: List of axes to use for similarity calculation
    :param procrustes: Whether to use procrustes alignment
    :return: Mean correlation coefficient and mean p-value
    """

    if axes_to_use is None:
        axes_to_use = [0, 1, 2]

    assert prediction.shape == target.shape, "The shape of prediction and target must match."

    target = target[:, :, axes_to_use]
    prediction = prediction[:, :, axes_to_use]

    if procrustes:
        target, prediction, _ = align_procrustes(target, prediction)

    correlations = []
    pvalues = []

    for keypoint in range(target.shape[1]):
        for coordinate in range(target.shape[2]):
            gt = target[:, keypoint, coordinate]
            pred = prediction[:, keypoint, coordinate]

            if np.std(gt) == 0 or np.std(pred) == 0:
                continue  # Skip if no variance in the data

            corr, pvalue = pearsonr(gt, pred)
            correlations.append(corr)
            pvalues.append(pvalue)

    if not correlations:
        return float('nan'), float('nan')  # Handle case where no valid correlations were calculated

    return np.mean(correlations), np.mean(pvalues)


def compute_CP(target, prediction, threshold=180, joints_to_use=None):
    """
    Compute the number of correct poses
    :param target: Ground truth 3D joint positions
    :param prediction: Predicted 3D joint positions
    :param threshold: Threshold for correct poses
    :param joints_to_use: List of joints to use for CP calculation
    :return: Number of correct poses
    """
    if joints_to_use is None:
        joints_to_use = list(range(target.shape[1]))

    assert target.shape == prediction.shape, "The shape of prediction and target must match."

    distances = np.linalg.norm(target[:, joints_to_use, :] - prediction[:, joints_to_use, :], axis=2)
    correct_poses = np.count_nonzero(distances < threshold, axis=1) == len(joints_to_use)

    return np.sum(correct_poses)


def compute_CPS(target, prediction, min_th=1, max_th=300, step=1,
                joints_to_use=None, procrustes=False):
    """
    Compute the correct pose score (CPS) according to (https://arxiv.org/abs/2011.14679) for different thresholds
    :param target: Ground truth 3D joint positions, shape [sample, joint, 3]
    :param prediction: Predicted 3D joint positions, shape [sample, joint, 3]
    :param min_th: Minimum threshold
    :param max_th: Maximum threshold
    :param step: Step size
    :param joints_to_use: List of joints to use for CPS calculation
    :param procrustes: Whether to use procrustes alignment
    :return: CPS
    :return: idx of best CPS
    """

    if joints_to_use is None:
        joints_to_use = list(range(target.shape[1]))

    assert len(prediction.shape) == len(target.shape), "The shape of prediction and target must match."

    cps_length = int((max_th + 1 - min_th) / step)
    thresholds = np.arange(min_th, max_th + 1, step)
    cps_best_list = np.zeros(cps_length, dtype=np.double)
    cp_values_list = np.empty((prediction.shape[0], len(thresholds)), dtype=np.double)

    if procrustes:
        target, prediction, err = align_procrustes(target, prediction)

    for i, threshold in enumerate(thresholds):
        cp_values_list[:, i] = compute_CP(target, prediction, threshold, joints_to_use)

    values = np.max(cp_values_list, axis=1)
    cps_idx = np.argmax(cp_values_list, axis=1)
    cps_best_list += np.sum(values)

    cps_best_list /= prediction.shape[0]
    cps_best = auc(thresholds, cps_best_list)

    return cps_best, cps_idx


def angle_2p_3d(joint_a, joint_b, joint_c):
    """
    Calculate the angle between two segments in 3D
    :param joint_a: [frames, joint, 3]
    :param joint_b: [frames, joint, 3]
    :param joint_c: [frames, joint, 3]
    :return joint_angles: [frames, 1]
    """
    # Vectorized calculation of vectors
    v1 = joint_a - joint_b
    v2 = joint_c - joint_b

    # Normalizing the vectors
    v1_norm = v1 / np.linalg.norm(v1, axis=2, keepdims=True)
    v2_norm = v2 / np.linalg.norm(v2, axis=2, keepdims=True)

    # Dot product and angle calculation
    dot_product = np.sum(v1_norm * v2_norm, axis=2)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Clip to handle numerical issues

    angles_rad = np.arccos(dot_product)
    angles_deg = np.degrees(angles_rad)

    return np.round(angles_deg, 2)


def angle_between_points(joint_a, joint_b, joint_c):
    """
    Calculate the angle between two segments in 3D
    :param joint_a: [frames, joint, 3]
    :param joint_b: [frames, joint, 3]
    :param joint_c: [frames, joint, 3]
    :return joint_angles: [frames, joint]
    """
    AB = joint_b - joint_a
    BC = joint_c - joint_b

    dot_product = np.einsum('...i,...i', AB, BC)
    magnitude_AB = np.linalg.norm(AB, axis=2)
    magnitude_BC = np.linalg.norm(BC, axis=2)

    angles_rad = np.arccos(np.clip(dot_product / (magnitude_AB * magnitude_BC), -1.0, 1.0))  # Clip to handle numerical issues
    angles_deg = np.degrees(angles_rad)

    return angles_deg


def orthogonal_projection(vector, normal):
    """
    Calculate the orthogonal projection of a vector onto a plane defined by a normal vector.
    :param vector: The vector to be projected, shape (n,)
    :param normal: The normal vector of the plane, shape (n,)
    :return: The orthogonal projection of the vector onto the plane, shape (n,)
    """
    proj = vector - np.dot(vector, normal) / np.dot(normal, normal) * normal
    return proj


def help_shoulder(hip_mid, shoulder_mid, shoulder_left, shoulder_right):
    """
    Calculate the help vector Ds = (hip_mid, shoulder_mid) x (shoulder_right, shoulder_left)
    :param hip_mid: ip center
    :param shoulder_mid: Shoulder center
    :param shoulder_left: Left shoulder
    :param shoulder_right: Right shoulder
    :return: Help vector Ds
    """
    ds = np.cross(hip_mid - shoulder_mid, shoulder_right - shoulder_left)
    return ds


def help_hip(Y, hip_left, hip_right):
    """
    Calculate the help vector Dh = Y x (hip_right, hip_left)
    :param Y: Y vector
    :param hip_left: Left hip
    :param hip_right: Right hip
    :return: Help vector Dh
    """
    dh = np.cross(Y, hip_right - hip_left)
    return dh


def shoulder_mid(shoulder_left, shoulder_right):
    """
    Calculate the shoulder mid-point
    :param shoulder_left: Left shoulder
    :param shoulder_right: Right shoulder
    :return: Shoulder mid-point
    """
    shoulder_mid_point = (shoulder_right + shoulder_left) / 2
    return shoulder_mid_point


def hip_mid(hip_left, hip_right):
    """
    Calculate the hip mid-point
    :param hip_left: Left hip
    :param hip_right: Right hip
    :return: Hip mid-point
    """
    hip_mid_point = (hip_right + hip_left) / 2
    return hip_mid_point


def trunk_angle(hip_mid, shoulder_mid, Ds):
    """
    Calculate the trunk angle
    :param hip_mid: Hip mid-point
    :param shoulder_mid: Shoulder mid-point
    :return: Trunk angle
    """
    trunk_angle_angle = 90 - angle_between_points(hip_mid, shoulder_mid, Ds)
    return trunk_angle_angle


def trunk_twist(hip_left, hip_right, shoulder_left, shoulder_right):
    """
    Calculate the trunk twist
    :param hip_left: Left hip
    :param hip_right: Right hip
    :param shoulder_left: Left shoulder
    :param shoulder_right: Right shoulder
    :return: Trunk twist
    """
    shoulder_mid_point = shoulder_mid(shoulder_left, shoulder_right)
    hip_mid_point = hip_mid(hip_left, hip_right)

    HH = hip_right - hip_left
    CB = shoulder_mid_point - hip_mid_point
    SS = shoulder_right - shoulder_left

    trunk_twist_angle = angle_between_points(orthogonal_projection(HH, CB), orthogonal_projection(SS, CB))
    return trunk_twist_angle


def trunk_bending(Y, CB, Dh):
    """
    Calculate the trunk bending
    :param Y: Y vector
    :param CB: CB vector
    :param Dh: Dh vector
    :return: Trunk bending
    """
    trunk_bending_angle = angle_between_points(Y, orthogonal_projection(CB, Dh))
    return trunk_bending_angle


def knee_angle(hip_left, knee_left, ankle_left):
    """
    Calculate the knee angle
    :param hip_left: Left hip
    :param knee_left: Left knee
    :param ankle_left: Left ankle
    :return: Knee angle
    """
    knee_angle_angle = angle_between_points(hip_left, knee_left, ankle_left)
    return knee_angle_angle


def shoulder_angle(elbow_left, shoulder_left, Ds, CB):
    """
    Calculate the shoulder angle
    :param elbow_left: Left elbow
    :param shoulder_left: Left shoulder
    :param Ds: Help vector Ds
    :return: Shoulder angle
    """
    ES = shoulder_left - elbow_left
    SE = elbow_left - shoulder_left

    shoulder_angle_angle = angle_between_points(orthogonal_projection(ES, np.cross(Ds, CB)),
                                                np.sign(SE * Ds))
    return shoulder_angle_angle


def elbow_angle(shoulder_left, elbow_left, wrist_left):
    """
    Calculate the elbow angle
    :param shoulder_left: Left shoulder
    :param elbow_left: Left elbow
    :param wrist_left: Left wrist
    :return: Elbow angle
    """
    elbow_angle_angle = angle_between_points(shoulder_left, elbow_left, wrist_left)
    return elbow_angle_angle


def ankle_angle(knee_left, ankle_left, toe_left):
    """
    Calculate the ankle angle
    :param knee_left: Left knee
    :param ankle_left: Left ankle
    :param toe_left: Left toe
    :return: Ankle angle
    """
    ankle_angle = angle_between_points(knee_left, ankle_left, toe_left)
    return ankle_angle


####################################################################
### Adjust according to https://www.mdpi.com/1424-8220/22/5/1729 ###
####################################################################
def calculate_mpsae(target, prediction, procrustes=True):
    """
    Calculate the mean per segment angle error.
    :param target: Ground truth 3D joint positions, dictionary with keys as joint names
    :param prediction: Predicted 3D joint positions, dictionary with keys as joint names
    :param joint_segments: List of joints to use for MPSAE calculation
    :return: Mean error, standard deviation of error, segment-wise errors
    """
    assert len(prediction) == len(target), "The shape of prediction and target must match."

    target_Y = np.array([0, 1, 0])
    prediction_Y = np.array([0, 1, 0])

    target, prediction, error_count = align_procrustes(target, prediction)

    target_hip_mid = hip_mid(target['hip_left'], target['hip_right'])
    prediction_hip_mid = hip_mid(prediction['hip_left'], prediction['hip_right'])
    target_shoulder_mid = shoulder_mid(target['shoulder_left'], target['shoulder_right'])
    prediction_shoulder_mid = shoulder_mid(prediction['shoulder_left'], prediction['shoulder_right'])
    target_CB = target_shoulder_mid - target_hip_mid
    prediction_CB = prediction_shoulder_mid - prediction_hip_mid
    target_Ds = help_shoulder(target_hip_mid, target_shoulder_mid, target['shoulder_left'], target['shoulder_right'])
    prediction_Ds = help_shoulder(prediction_hip_mid, prediction_shoulder_mid, prediction['shoulder_left'],
                                  prediction['shoulder_right'])
    target_Dh = help_hip(target_Y, target['hip_left'], target['hip_right'])
    prediction_Dh = help_hip(prediction_Y, prediction['hip_left'], prediction['hip_right'])

    trunk_twist_error = np.abs(
        trunk_twist(target['hip_left'], target['hip_right'], target['shoulder_left'], target['shoulder_right']) -
        trunk_twist(prediction['hip_left'], prediction['hip_right'], prediction['shoulder_left'],
                    prediction['shoulder_right']))

    trunk_bending_error = np.abs(
        trunk_bending(target_Y, target_CB, target_Dh) -
        trunk_bending(prediction_Y, prediction_CB, prediction_Dh))

    trunk_angle_error = np.abs(
        trunk_angle(target_hip_mid, target_shoulder_mid, target_Ds) -
        trunk_angle(prediction_hip_mid, prediction_shoulder_mid, prediction_Ds))

    knee_left_error = np.abs(
        knee_angle(target['hip_left'], target['knee_left'], target['ankle_left']) -
        knee_angle(prediction['hip_left'], prediction['knee_left'], prediction['ankle_left']))

    knee_right_error = np.abs(
        knee_angle(target['hip_right'], target['knee_right'], target['ankle_right']) -
        knee_angle(prediction['hip_right'], prediction['knee_right'], prediction['ankle_right']))

    shoulder_left_error = np.abs(
        shoulder_angle(target['elbow_left'], target['shoulder_left'], target_Ds, target_CB) -
        shoulder_angle(prediction['elbow_left'], prediction['shoulder_left'], prediction_Ds, prediction_CB))

    shoulder_right_error = np.abs(
        shoulder_angle(target['elbow_right'], target['shoulder_right'], target_Ds, target_CB) -
        shoulder_angle(prediction['elbow_right'], prediction['shoulder_right'], prediction_Ds, prediction_CB))

    elbow_left_error = np.abs(
        elbow_angle(target['shoulder_left'], target['elbow_left'], target['wrist_left']) -
        elbow_angle(prediction['shoulder_left'], prediction['elbow_left'], prediction['wrist_left']))

    elbow_right_error = np.abs(
        elbow_angle(target['shoulder_right'], target['elbow_right'], target['wrist_right']) -
        elbow_angle(prediction['shoulder_right'], prediction['elbow_right'], prediction['wrist_right']))

    ankle_left_error = np.abs(
        ankle_angle(target['knee_left'], target['ankle_left'], target['toe_left']) -
        ankle_angle(prediction['knee_left'], prediction['ankle_left'], prediction['toe_left']))

    ankle_right_error = np.abs(
        ankle_angle(target['knee_right'], target['ankle_right'], target['toe_right']) -
        ankle_angle(prediction['knee_right'], prediction['ankle_right'], prediction['toe_right']))

    single_segment_error = np.array(
        [trunk_twist_error, trunk_bending_error, trunk_angle_error, knee_left_error, knee_right_error,
         shoulder_left_error, shoulder_right_error, elbow_left_error, elbow_right_error, ankle_left_error,
         ankle_right_error])

    single_segment_mean_error = np.mean(single_segment_error, axis=0)
    single_segment_std_error = np.std(single_segment_error, axis=0)

    # Calculate R2 for each segment
    segment_r2 = calculate_r2(single_segment_error, np.zeros_like(single_segment_error))

    return single_segment_mean_error, single_segment_std_error, single_segment_error, segment_r2


def calculate_r2(target, predicted):
    """
    Calculate the coefficient of determination R^2
    :param target: Actual values
    :param predicted: Predicted values
    :return: R^2
    """
    ss_res = np.sum((target - predicted) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def plot_r2(values, segment_names):
    """
    Plot R^2 values for each segment
    :param values: list of R^2 values
    :param segment_names: names of the segments
    """
    plt.figure(figsize=(10, 5))
    plt.bar(segment_names, values, color='skyblue')
    plt.xlabel('Segment')
    plt.ylabel('R^2 Value')
    plt.title('R^2 for each Segment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def calculate_mpsae_old(target, prediction, joint_segments):
    """
    Calculate the mean per segment angle error. Target and prediction are dictionaries with the same keys segments is a list of lists of segment names.
    :param target: [frames, joint, 3]
    :param prediction: [frames, joint, 3]
    :param segments: list of segment names and their indexes
    :return: mean, std, segment_errors_mean, segment_errors_std
    """

    assert len(prediction.shape) == len(target.shape)

    angle_errors = []
    segment_errors_mean = {}
    segment_errors_std = {}
    for segment_name, segment_array in joint_segments.items():
        angle_target = angle_2p_3d(target[:, segment_array[0]], target[:, segment_array[1]],
                                   target[:, segment_array[2]])  # batch, 1
        angle_prediction = angle_2p_3d(prediction[:, segment_array[0]], prediction[:, segment_array[1]],
                                       prediction[:, segment_array[2]])
        angle_err = np.abs(np.array(angle_target) - np.array(angle_prediction))
        angle_err = np.round(angle_err, 2)
        angle_errors.append(angle_err)  # segments, batch, 1
        segment_errors_mean[segment_name] = np.mean(angle_err)
        segment_errors_std[segment_name] = np.std(angle_err)
    single_segment_err = np.mean(np.array(angle_errors), axis=1)

    return np.mean(single_segment_err), np.std(single_segment_err), segment_errors_mean, segment_errors_std


def calculate_symmetry_error(prediction, bones, bone_pairs):
    """
    Calculate the symmetry error for a given set of bones
    :param prediction: Predicted 3D joint positions
    :param bones: Definition of bones
    :param bone_pairs: Pairs of left and right bones
    :return:
    """
    # Extract bones as delta positions and calculate length
    bone_start = [segment_array[0] for segment_array in bones.values()]
    bone_end = [segment_array[1] for segment_array in bones.values()]

    bone_lengths = np.linalg.norm(prediction[:, bone_start, :] - prediction[:, bone_end, :], axis=2)

    # Find matching bones in skeleton
    idx_bone0 = [segment_array[0] for segment_array in bone_pairs.values()]
    idx_bone1 = [segment_array[1] for segment_array in bone_pairs.values()]
    bone0 = bone_lengths[:, idx_bone0]
    bone1 = bone_lengths[:, idx_bone1]

    # Calculate the absolute length difference between symmetries
    absolute_error = np.abs(bone0 - bone1)

    # Calculate the average error for all bones
    single_bone_err = np.mean(absolute_error, axis=0)
    mean = np.mean(single_bone_err)
    std = np.std(single_bone_err)

    return mean, std, single_bone_err


def wrap_to_pi(x):
    xwrap = np.remainder(x, 2 * np.pi)
    mask = np.abs(xwrap) > np.pi
    xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
    mask1 = x < 0
    mask2 = np.remainder(x, np.pi) == 0
    mask3 = np.remainder(x, 2 * np.pi) != 0
    xwrap[mask1 & mask2 & mask3] -= 2 * np.pi
    return xwrap


def calculate_mpjphe(target, prediction):
    # calculate the mean per joint phase error
    # target and prediction are dictionaries with the same keys [batch, joint, 3]

    assert len(prediction.shape) == len(target.shape)

    # padding to remove edge effects (DOI: 10.1123/jab.2018-0396)
    pad_width = np.sqrt(target.shape[0]).astype(int) + 1
    pad_width = ((pad_width ** 2 - target.shape[0]) / 2).astype(int)
    target = np.pad(target, ((pad_width, pad_width), (0, 0), (0, 0)), 'constant', constant_values=0)
    prediction = np.pad(prediction, ((pad_width, pad_width), (0, 0), (0, 0)), 'constant', constant_values=0)

    # calculate Hilbert transform
    h_target = signal.hilbert(target, axis=0)
    h_prediction = signal.hilbert(prediction, axis=0)

    # calculate phase
    phase_target = np.unwrap(np.angle(h_target))
    phase_prediction = np.unwrap(np.angle(h_prediction))

    # calculate phase difference
    phase_diff = wrap_to_pi(phase_target - phase_prediction)

    # remove padding
    phase_diff = phase_diff[pad_width:-pad_width]

    return np.median(phase_diff), np.std(phase_diff), phase_target, phase_prediction, phase_diff

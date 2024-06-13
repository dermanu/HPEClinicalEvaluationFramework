import numpy as np
from scipy import signal, stats
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import wandb


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

    mpjpe = np.linalg.norm(prediction - target, axis=1)
    mean = np.mean(mpjpe)
    std = np.std(mpjpe)

    return mean, std


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
    often referred to as "Protocol #2" in many papers..
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


def mean_velocity_error(prediction, target, sample_rate, procrustes=False):
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
        mean, std = calculate_mpjpe(velocity_target, velocity_predicted)

    return mean, std


def mean_acceleration_error(prediction, target, sample_rate, procrustes=False):
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


def calculate_correlation(target, prediction, axes_to_use=None, procrustes=False):
    """
    Calculate average Pearson correlation coefficient for all joints and axes to measure signal similarity.
    :param target: Ground truth 3D joint positions, shape [sample, 3]
    :param prediction: Predicted 3D joint positions, shape [sample, 3]
    :param axes_to_use: List of axes to use for similarity calculation
    :param procrustes: Whether to use procrustes alignment
    :return: Mean correlation coefficient and mean p-value
    """

    if axes_to_use is None:
        axes_to_use = [0, 1, 2]

    assert prediction.shape == target.shape, "The shape of prediction and target must match."

    target = target[:, axes_to_use]
    prediction = prediction[:, axes_to_use]

    if procrustes:
        target, prediction, _ = align_procrustes(target, prediction)

    correlations = []
    pvalues = []

    for coordinate in range(target.shape[1]):
        gt = target[:, coordinate]
        pred = prediction[:, coordinate]

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


def calculate_angle(joint_a, joint_b, joint_c):
    """
    Calculate the angle formed at joint_b by the lines connecting joint_a and joint_c
    :param joint_a: 3D coordinates of joint A, numpy array of shape [sample, joint, 3]
    :param joint_b: 3D coordinates of joint B (vertex of the angle), numpy array of shape [sample, joint, 3]
    :param joint_c: 3D coordinates of joint C, numpy array of shape [sample, joint, 3]
    :return: Angles in degrees, numpy array of shape [sample, joint]
    """
    v1 = joint_a - joint_b
    v2 = joint_c - joint_b
    v1_norm = v1 / np.linalg.norm(v1, axis=2, keepdims=True)
    v2_norm = v2 / np.linalg.norm(v2, axis=2, keepdims=True)
    dot_product = np.sum(v1_norm * v2_norm, axis=2)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Clip to ensure they are within the valid range for arccos [-1, 1]
    angles_rad = np.arccos(dot_product)
    angles_deg = np.degrees(angles_rad)
    return np.round(angles_deg, 2)


def orthogonal_projection(vector, normal):
    """
    Calculate the orthogonal projection of a vector onto a plane defined by a normal vector.
    :param vector: The vector to be projected, shape (n,)
    :param normal: The normal vector of the plane, shape (n,)
    :return: The orthogonal projection of the vector onto the plane, shape (n,)
    """
    return vector - np.dot(vector, normal) / np.dot(normal, normal) * normal


def get_vertical_axis_from_calibration(hip_left, hip_right, shoulder_left, shoulder_right):
    """
    Determine the vertical axis based on the positions of the hips and shoulders. Needs a relative neutral pose
    :param hip_left: numpy array of shape (3,), position of the left hip
    :param hip_right: numpy array of shape (3,), position of the right hip
    :param shoulder_left: numpy array of shape (3,), position of the left shoulder
    :param shoulder_right: numpy array of shape (3,), position of the right shoulder
    :return: normalized vertical axis vector
    """
    hip_mid = (hip_left + hip_right) / 2
    shoulder_mid = (shoulder_left + shoulder_right) / 2
    vertical_vector = shoulder_mid - hip_mid
    return vertical_vector / np.linalg.norm(vertical_vector)


####################################################################
### Adjust according to https://www.mdpi.com/1424-8220/22/5/1729 ###
####################################################################
def calculate_joint_angles(keypoints, Y_vector=np.array([0, 1, 0])):
    """
    Calculate various joint angles based on 3D keypoints.
    :param keypoints: Dictionary of 3D coordinates for each joint
    :param Y_vector: Reference vertical axis vector (default is the Y-axis)
    :return: Dictionary of calculated joint angles
    """
    angles = {}

    # Midpoints
    shoulder_mid = (keypoints['right_shoulder'] + keypoints['left_shoulder']) / 2
    hip_mid = (keypoints['right_hip'] + keypoints['left_hip']) / 2

    # Help vectors
    D_s = np.cross(hip_mid - shoulder_mid, keypoints['right_shoulder'] - keypoints['left_shoulder'])
    D_h = np.cross(Y_vector, keypoints['right_hip'] - keypoints['left_hip'])

    # Trunk angles
    angles['trunk_angle'] = 90 - calculate_angle(hip_mid - shoulder_mid, D_h)
    angles['trunk_twist'] = calculate_angle(
        orthogonal_projection(keypoints['left_hip'] - keypoints['right_hip'], shoulder_mid - hip_mid),
        orthogonal_projection(keypoints['right_shoulder'] - keypoints['left_shoulder'], shoulder_mid - hip_mid))
    angles['trunk_bend'] = calculate_angle(Y_vector, orthogonal_projection(shoulder_mid - hip_mid, D_h))

    # Lower limb angles
    angles['knee_angle_l'] = calculate_angle(keypoints['left_hip'], hip_mid, keypoints['left_ankle']) # HK?
    angles['ankle_angle_l'] = calculate_angle(keypoints['left_knee'], keypoints['left_ankle'], keypoints['left_toe'])
    angles['knee_angle_r'] = calculate_angle(keypoints['right_hip'], hip_mid, keypoints['right_ankle'])
    angles['ankle_angle_r'] = calculate_angle(keypoints['right_knee'], keypoints['right_ankle'], keypoints['right_toe'])

    # Upper limb angles
    angles['shoulder_angle_l'] = calculate_angle(
        orthogonal_projection(keypoints['left_elbow'] - keypoints['left_shoulder'], np.cross(D_s, shoulder_mid - hip_mid)),
        shoulder_mid - hip_mid) * np.sign(np.dot(keypoints['left_elbow'] - keypoints['left_shoulder'], D_s))
    angles['elbow_angle_l'] = calculate_angle(keypoints['left_shoulder'], keypoints['left_elbow'], keypoints['left_wrist'])
    angles['shoulder_angle_r'] = calculate_angle(
        orthogonal_projection(keypoints['right_elbow'] - keypoints['right_shoulder'], np.cross(D_s, shoulder_mid - hip_mid)),
        shoulder_mid - hip_mid) * np.sign(np.dot(keypoints['right_elbow'] - keypoints['right_shoulder'], D_s))
    angles['elbow_angle_r'] = calculate_angle(keypoints['right_shoulder'], keypoints['right_elbow'], keypoints['right_wrist'])

    return angles


def calculate_angle_error(target, prediction, Y_target=np.array([0, 1, 0]), Y_prediction=np.array([0, 1, 0]),
                          procrustes=False, r2=True, box=True):
    """
    Calculate the angle error between target and prediction joint positions.
    :param target: Ground truth 3D joint positions, shape [sample, joint, 3]
    :param prediction: Predicted 3D joint positions, shape [sample, joint, 3]
    :param Y_target: Reference vertical axis vector for target (default is the Y-axis)
    :param Y_prediction: Reference vertical axis vector for prediction (default is the Y-axis)
    :param procrustes: Whether to use Procrustes alignment
    :param r2: Whether to calculate R² value
    :param box: Whether to plot a box plot of the angle errors
    :return: Mean error, standard deviation of error, and R² value (if calculated)
    """
    if procrustes:
        target, prediction, error_count = align_procrustes(target, prediction)
    target_angles = calculate_joint_angles(target, Y_target)
    prediction_angles = calculate_joint_angles(target, Y_prediction)
    angle_error = np.abs(target_angles - prediction_angles)
    mean_error = np.mean(angle_error)
    std_error = np.std(angle_error)

    r2 = None
    if r2:
        r2 = plot_calculate_r2(target_angles, prediction_angles)

    if box:
        plot_box(angle_error)
    return mean_error, std_error, r2


def plot_calculate_r2(target_angles_dict, prediction_angles_dict):
    """
    Plot R² values for the linear regression fit of target vs prediction angles for each metric and calculate R² value.
    :param target_angles_dict: Dictionary where keys are metric names and values are lists of ground truth angles
    :param prediction_angles_dict: Dictionary where keys are metric names and values are lists of predicted angles
    :return: Dictionary of R² values for each metric
    """
    r2_values = {}

    fig = go.Figure()

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'black', 'pink']
    symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right', 'pentagon', 'hexagon']

    for i, metric in enumerate(target_angles_dict.keys()):
        target_angles = np.array(target_angles_dict[metric]).reshape(-1, 1)
        prediction_angles = np.array(prediction_angles_dict[metric])
        reg = LinearRegression().fit(target_angles, prediction_angles)
        prediction_line = reg.predict(target_angles)
        r2 = r2_score(prediction_angles, prediction_line)
        r2_values[metric] = r2

        # Scatter plot of the actual values with different colors and marker forms
        fig.add_trace(go.Scatter(
            x=target_angles.flatten(),
            y=prediction_angles,
            mode='markers',
            name=f'{metric} Actual values',
            marker=dict(color=colors[i % len(colors)], symbol=symbols[i % len(symbols)])
        ))

        # Linear regression line
        fig.add_trace(go.Scatter(
            x=target_angles.flatten(),
            y=prediction_line,
            mode='lines',
            name=f'{metric} Linear fit (R²={r2:.2f})',
            line=dict(color=colors[i % len(colors)])
        ))

    fig.update_layout(
        title='Prediction Angles vs Target Angles',
        xaxis_title='Target Angles',
        yaxis_title='Prediction Angles',
        legend_title='Legend',
        showlegend=True
    )

    wandb.log({"R2 plot": fig})
    return r2_values


def plot_box(errors_dict):
    """
    Plot different errors in a box plot.
    :param errors_dict: Dictionary where keys are metric names and values are lists of error values.
    """
    fig = go.Figure()

    # Different colors and symbols for different metrics
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'black', 'pink']
    symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right', 'pentagon', 'hexagon']

    for i, (metric, errors) in enumerate(errors_dict.items()):
        fig.add_trace(go.Box(y=errors, name=metric, boxmean=True))

    fig.update_layout(
        title='Error Metrics',
        xaxis_title='Metrics',
        yaxis_title='Error Values',
        showlegend=True
    )

    wandb.log({"Angular error box": fig})


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
        angle_target = calculate_angle(target[:, segment_array[0]], target[:, segment_array[1]],
                                   target[:, segment_array[2]])  # batch, 1
        angle_prediction = calculate_angle(prediction[:, segment_array[0]], prediction[:, segment_array[1]],
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

"""
Collection of evaluation metrics. Not all are tested.
"""

import numpy as np
from scipy import signal
from sklearn.metrics import auc
from scipy.stats import pearsonr
from scipy.spatial import procrustes
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import wandb
import pingouin as pg
import pandas as pd
from scipy.signal import savgol_filter


def calculate_mpjpe(target, prediction, axis=1):
    """
    Mean per-joint position error (MPJPE)

    Parameters:
    - target: Ground truth 3D joint positions
    - prediction: Predicted 3D joint positions

    Returns:
    - Mean and standard deviation of the MPJPE
    """
    assert prediction.shape == target.shape, "The shape of prediction and target must match."

    mpjpe = np.linalg.norm(prediction - target, axis=axis)

    mpjpe = mpjpe[~np.isnan(mpjpe)]
    mean = np.mean(mpjpe)
    std = np.std(mpjpe)
    Q1, median, Q3 = np.percentile(mpjpe, [25, 50, 75])
    IQR = Q3 - Q1
    loval = Q1 - 1.5 * IQR
    hival = Q3 + 1.5 * IQR
    wiskhi = np.compress(mpjpe <= hival, mpjpe)
    wisklo = np.compress(mpjpe >= loval, mpjpe)
    actual_hival = np.max(wiskhi)
    actual_loval = np.min(wisklo)

    return mean, std, [Q1, median, Q3, loval, hival, actual_loval, actual_hival]

def align_procrustes(target, prediction):
    """
    Procrustes MPJPE: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    Based on the implementation from https://github.com/miraymen/3dpw-eval/blob/master/evaluate.py

    Parameters:
    - target: Ground truth 3D joint positions, shape [sample, joint, 3]
    - prediction: Predicted 3D joint positions, shape [sample, joint, 3]

    Returns:
    - gt_all: Ground truth 3D joint positions after alignment, shape [sample, joint, 3]
    - pred_all: Predicted 3D joint positions after alignment, shape [sample, joint, 3]
    - error_count: Error count of failed alignments
    """
    mtx1_3d, mtx2_3d, disparity_3d = procrustes(target, prediction)

    return mtx1_3d, mtx2_3d


def calculate_pmpjpe(target, prediction, procrustes=True, axis=1):
    """
    Procrustes MPJPE: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.

    Parameters:
    - target: Ground truth 3D joint positions
    - prediction: Predicted 3D joint positions

    Returns:
    - Mean and standard deviation of the PMPJPE
    - Error count of failed alignments
    """
    assert prediction.shape == target.shape, "The shape of prediction and target must match."

    if procrustes:
        target, prediction, error_count = align_procrustes(target, prediction)
    else:
        error_count = 0
    mean, std, Qs = calculate_mpjpe(target, prediction, axis=axis)

    return mean, std, error_count, Qs

################################
## PCK not extensively tested ##
################################
def calculate_pck(target, prediction, threshold=100.0,
                  joints_to_use=None, procrustes=False):
    """
    Calculate percentage of correct keypoints (PCK) in [%] (https://arxiv.org/pdf/1611.09813.pdf)

    Parameters:
    - target: Ground truth 3D joint positions
    - prediction: Predicted 3D joint positions
    - threshold: Threshold for correct keypoints
    - joints_to_use: List of joints to use for PCK calculation
    - procrustes: Whether to use procrustes alignment

    Returns:
    - PCK in [%]
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

#############################
## Not really used anymore ##
#############################
def mean_velocity_error(target, prediction, sample_rate,
                        procrustes=False,
                        smooth=False, window_length=25, polyorder=3):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)

    Parameters:
    - prediction: Predicted 3D joint positions in mm (shape: [frames, joints, 3])
    - target: Ground truth 3D joint positions in mm (shape: [frames, joints, 3])
    - sample_rate: Sample rate in Hz
    - procrustes: Whether Procrustes alignment should be used
    - axis: Axis for the norm calculation (default: 2 -> across x,y,z)
    - smooth: Whether to apply Savitzky–Golay smoothing to raw positions
    - window_length: Window length for the Savitzky–Golay filter
    - polyorder: Polynomial order for the Savitzky–Golay filter

    Returns:
    - (mean_error [m/s], std_error [m/s], [Q1, median, Q3, loval, hival, actual_loval, actual_hival])
    """
    assert prediction.shape == target.shape, "Shape mismatch between prediction and target."

    if smooth:
        # Apply smoothing along the time dimension (axis=0)
        prediction = savgol_filter(prediction, window_length=window_length,
                                   polyorder=polyorder, axis=0)
        #target = savgol_filter(target, window_length=window_length,
        #                       polyorder=polyorder, axis=0)

    # Needs to be changed to central difference
    #velocity_predicted = np.diff(prediction, axis=0) * sample_rate
    #velocity_target = np.diff(target, axis=0) * sample_rate
    velocity_predicted = (prediction[1:] - prediction[:-1]) / (1/sample_rate)
    velocity_target = (target[1:] - target[:-1]) / (1/sample_rate)

    true_velocity = np.mean(np.mean(np.abs(velocity_target), axis=(1, 2)))
    gt_velocity = np.mean(np.abs(velocity_target), axis=(1, 2))
    predicted_velocity = np.mean(np.abs(velocity_predicted), axis=(1, 2))

    # Create valid mask for NaNs
    valid_mask = ~np.isnan(velocity_predicted) & ~np.isnan(velocity_target)

    if procrustes:
        # Replace NaNs with zeros temporarily for alignment
        velocity_predicted_filled = np.where(valid_mask, velocity_predicted, 0)
        velocity_target_filled = np.where(valid_mask, velocity_target, 0)
        mean, std, err = calculate_pmpjpe(velocity_target_filled, velocity_predicted_filled)
        Q1 = Q3 = median = loval = hival = actual_loval = actual_hival = 0
    else:
        #mpjpe = np.linalg.norm(velocity_predicted - velocity_target, axis=axis)
        mpjve = np.mean(np.abs(velocity_target - velocity_predicted), axis=(1, 2))
        mpjve = mpjve[~np.isnan(mpjve)]

        # Calculate statistics
        mean = np.mean(mpjve)
        std = np.std(mpjve)
        Q1, median, Q3 = np.percentile(mpjve, [25, 50, 75])
        IQR = Q3 - Q1
        loval = (Q1 - 1.5 * IQR)
        hival = (Q3 + 1.5 * IQR)

        # Whiskers for outlier detection
        wiskhi = np.compress(mpjve <= hival, mpjve)
        wisklo = np.compress(mpjve >= loval, mpjve)
        actual_hival = np.max(wiskhi)
        actual_loval = np.min(wisklo)

    # Return mean & std in m/s plus additional percentile info
    return mean, std, [Q1, median, Q3, loval, hival, actual_loval, actual_hival]

###############################################
## Acceleration error not extensively tested ##
###############################################
def mean_acceleration_error(prediction, target, sample_rate, procrustes=False):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    Parameters:
    - prediction: Predicted 3D joint positions
    - target: Ground truth 3D joint positions
    - procrustes: Whether to use procrustes alignment
    - sample_rate: Sample rate in Hz
    - procrustes: Weather Procrustes alignment should be used

    Returns:
    - Mean and standard deviation of the mean per-joint velocity error
    """

    assert prediction.shape == target.shape, "The shape of prediction and target must match."

    if procrustes:
        target, prediction, err = align_procrustes(target, prediction)

    # Use central difference and smoothing
    acceleration_predicted = np.diff(np.diff(prediction, axis=0), axis=0) * sample_rate
    acceleration_target = np.diff(np.diff(target, axis=0), axis=0) * sample_rate

    mean, std, err = calculate_pmpjpe(acceleration_target, acceleration_predicted)

    return mean/1000, std/1000


def calculate_correlation(target, prediction, axes_to_use=None, procrustes=False):
    """
    Calculate average Pearson correlation coefficient (PCC) for all joints and axes to measure signal similarity.

    Parameters:
    - target: Ground truth 3D joint positions, shape [sample, 3]
    - prediction: Predicted 3D joint positions, shape [sample, 3]
    - axes_to_use: List of axes to use for similarity calculation
    - procrustes: Whether to use procrustes alignment

    Returns:
    - Mean correlation coefficient and mean p-value
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

        valid_mask = ~np.isnan(gt) & ~np.isnan(pred)
        gt_valid = gt[valid_mask]
        pred_valid = pred[valid_mask]

        if len(gt_valid) < 2:
            # Not enough data to compute correlation
            continue

        if np.std(gt_valid) == 0 or np.std(pred_valid) == 0:
            print('No variation in the data for the correlation')
            continue

        corr, pvalue = pearsonr(gt_valid, pred_valid)
        correlations.append(corr)
        pvalues.append(pvalue)

    if not correlations:
        return float('nan'), float('nan')  # Handle case where no valid correlations were calculated

    return np.nanmean(correlations), np.nanmean(pvalues)


####################################################
## Number of correct poses not extensively tested ##
####################################################
def compute_CP(target, prediction, threshold=180, joints_to_use=None):
    """
    Compute the number of correct poses
    Parameters:
    - target: Ground truth 3D joint positions
    - prediction: Predicted 3D joint positions
    - threshold: Threshold for correct poses
    - joints_to_use: List of joints to use for CP calculation

    Returns:
    - Number of correct poses
    """
    if joints_to_use is None:
        joints_to_use = list(range(target.shape[1]))

    assert target.shape == prediction.shape, "The shape of prediction and target must match."

    distances = np.linalg.norm(target[:, joints_to_use, :] - prediction[:, joints_to_use, :], axis=2)
    correct_poses = np.count_nonzero(distances < threshold, axis=1) == len(joints_to_use)

    return np.sum(correct_poses)


####################################################
## Symmetry not extensively tested ##
####################################################
def calculate_symmetry_error(prediction, bones, bone_pairs):
    """
    Calculate the symmetry error for a given set of bones

    Parameters:
    - prediction: Predicted 3D joint positions
    - bones: Definition of bones
    - bone_pairs: Pairs of left and right bones

    Returns:
    - Overall mean and std, and single bone symmetry error
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


def calculate_sem_mdc(sd, reliability, z_value=1.96):
    """nd Minimal Detectable Change (MDC).

    Calculate the Standard Error of Measurement (SEM) a
    :param sd: Standard deviation of the measurements
    :param reliability: Intraclass Correlation Coefficient (ICC) or other reliability metric
    :param z_value: Z value for the desired confidence level (default 1.96 for 95% confidence)
    :return: SEM and MDC
    """
    sem = sd * np.sqrt(1 - reliability)
    mdc = sem * z_value * np.sqrt(2)
    return sem, mdc


def calculate_icc(gt_data, pred_data):
    # Create a DataFrame for the ICC calculation
    df = pd.DataFrame({
        'subject': np.repeat(np.arange(len(gt_data)), 2),
        'measurement': np.tile(['gt', 'pred'], len(gt_data)),
        'score': np.concatenate([gt_data, pred_data])
    })

    # Calculate ICC
    icc_results = pg.intraclass_corr(data=df, targets='subject', raters='measurement', ratings='score')

    # Extract the ICC value, and the lower and upper bounds for ICC(A,1)
    icc_a1 = icc_results.loc[(icc_results['Type'] == 'ICC3'), 'ICC'].values[0]
    icc_lb = icc_results.loc[(icc_results['Type'] == 'ICC3'), 'CI95%'].values[0][0]
    icc_up = icc_results.loc[(icc_results['Type'] == 'ICC3'), 'CI95%'].values[0][1]

    return icc_a1, icc_lb, icc_up


def calculate_bias(gt_data, pred_data):
    return np.mean(pred_data - gt_data, axis=0)


def calculate_pearson_r(gt_data, pred_data):
    r_values, p_values = [], []
    r, p = pearsonr(gt_data, pred_data)
    r_values.append(r)
    p_values.append(p)
    return np.array(r_values), np.array(p_values)

################################################
## Correct poses score not extensively tested ##
################################################
def compute_CPS(target, prediction, min_th=1, max_th=300, step=1,
                joints_to_use=None, procrustes=False):
    """
    Compute the correct pose score (CPS) according to (https://arxiv.org/abs/2011.14679) for different thresholds

    Parameters:
    - target: Ground truth 3D joint positions, shape [sample, joint, 3]
    - prediction: Predicted 3D joint positions, shape [sample, joint, 3]
    - min_th: Minimum threshold
    - max_th: Maximum threshold
    - step: Step size
    - joints_to_use: List of joints to use for CPS calculation
    - procrustes: Whether to use procrustes alignment

    Returns:
    - CPS
    - idx of best CPS
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


####################################################################################################################################
## Old way of calculating the joint angles. Not correct. Should be replaced with angle_metrics_tpose.py in evaluation_pipeline.py ##
####################################################################################################################################
def calculate_angle_error(target, prediction, Y_target=np.array([0, 1, 0]), Y_prediction=np.array([0, 1, 0]),
                          procrustes=False, calculate_r2=True, box=True):
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
    prediction_angles = calculate_joint_angles(prediction, Y_prediction)
    angle_names = list(target_angles.keys())

    target_angle_values = []
    prediction_angle_values = []
    for joint, angle in target_angles.items():
        target_angle_values.append(angle)
    for joint, angle in prediction_angles.items():
        prediction_angle_values.append(angle)

    angle_error = np.abs(np.array(target_angle_values) - np.array(prediction_angle_values))
    mean_error = np.mean(angle_error, axis=1)
    std_error = np.std(angle_error, axis=1)

    mean_dict = {angle: (mean_error[i]) for i, angle in enumerate(angle_names)}
    std_dict = {angle: (std_error[i]) for i, angle in enumerate(angle_names)}

    r2 = None
    if calculate_r2:
        r2 = plot_calculate_r2(target_angles, prediction_angles)

    if box:
        angle_error = {angle: (angle_error[i]) for i, angle in enumerate(angle_names)}
        plot_box(angle_error)
    return mean_dict, std_dict, r2

def calculate_angle_point(joint_a, joint_b, joint_c):
    """
    Calculate the angle formed at joint_b by the lines connecting joint_a and joint_c
    :param joint_a: 3D coordinates of joint A, numpy array of shape [sample, joint, 3]
    :param joint_b: 3D coordinates of joint B (vertex of the angle), numpy array of shape [sample, joint, 3]
    :param joint_c: 3D coordinates of joint C, numpy array of shape [sample, joint, 3]
    :return: Angles in degrees, numpy array of shape [sample, joint]
    """
    # Check if any inputs need reshaping
    if len(joint_a.shape) == 1:
        joint_a = joint_a.reshape(-1, joint_a.shape[0], 3)
    if len(joint_b.shape) == 1:
        joint_b = joint_b.reshape(-1, joint_b.shape[0], 3)
    if len(joint_c.shape) == 1:
        joint_c = joint_c.reshape(-1, joint_c.shape[0], 3)

    v1 = joint_a - joint_b
    v2 = joint_c - joint_b
    v1_norm = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2_norm = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    dot_product = np.sum(v1_norm * v2_norm, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Clip to ensure they are within the valid range for arccos [-1, 1]
    angles_rad = np.arccos(dot_product)
    angles_deg = np.degrees(angles_rad)
    return np.round(angles_deg, 2)


def orthogonal_projection(vector, normal):
    """
    Calculate the orthogonal projection of a vector onto a plane defined by a normal vector.
    :param vector: The vector to be projected, shape (batch, 3)
    :param normal: The normal vector of the plane, shape (batch, 3)
    :return: The orthogonal projection of the vector onto the plane, shape (batch, 3)
    """
    dot_product = np.einsum('ij,ij->i', vector, normal)
    normal_norm_sq = np.einsum('ij,ij->i', normal, normal)
    projection = vector - (dot_product / normal_norm_sq)[:, np.newaxis] * normal
    return projection


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


def calculate_angle_vector(v1, v2):
    """
    Calculate the angle between two vectors.
    :param v1: First vector, numpy array of shape (batch, 3)
    :param v2: Second vector, numpy array of shape (batch, 3)
    :return: Angles in degrees, numpy array of shape (batch,)
    """
    # Check if v1 needs reshaping
    if len(v1.shape) == 1:
        v1 = v1.reshape(-1, 3)

    v1_norm = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2_norm = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    dot_product = np.einsum('ij,ij->i', v1_norm, v2_norm)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Clip to ensure they are within the valid range for arccos [-1, 1]
    angles_rad = np.arccos(dot_product)
    angles_deg = np.degrees(angles_rad)
    return np.round(angles_deg, 2)


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
    hip_mid = hip_mid.reshape(-1, 3)
    shoulder_mid = shoulder_mid.reshape(-1, 3)

    # Help vectors
    D_s = np.cross(hip_mid - shoulder_mid, keypoints['right_shoulder'] - keypoints['left_shoulder'])
    D_h = np.cross(Y_vector, keypoints['right_hip'] - keypoints['left_hip'])

    # Trunk angles
    angles['trunk_angle'] = 90 - calculate_angle_vector(hip_mid - shoulder_mid, D_h)
    angles['trunk_twist'] = 180 - calculate_angle_vector(
        orthogonal_projection(keypoints['left_hip'] - keypoints['right_hip'], shoulder_mid - hip_mid),
        orthogonal_projection(keypoints['right_shoulder'] - keypoints['left_shoulder'], shoulder_mid - hip_mid))
    angles['trunk_bend'] = 90 - calculate_angle_vector(Y_vector, orthogonal_projection(shoulder_mid - hip_mid, D_h))

    # Lower limb angles
    angles['knee_angle_l'] = 90 + calculate_angle_point(keypoints['left_hip'], hip_mid, keypoints['left_ankle']) # Really mid hip here?
    angles['knee_angle_r'] = 90 + calculate_angle_point(keypoints['right_hip'], hip_mid, keypoints['right_ankle'])  # Really mid hip here?
    angles['ankle_angle_l'] = calculate_angle_point(keypoints['left_knee'], keypoints['left_ankle'], keypoints['left_foot_index'])
    angles['ankle_angle_r'] = calculate_angle_point(keypoints['right_knee'], keypoints['right_ankle'], keypoints['right_foot_index'])

    # Upper limb angles
    angles['shoulder_side_l'] = calculate_angle_vector(orthogonal_projection(keypoints['left_elbow'] - keypoints['left_shoulder'], np.cross(D_s, shoulder_mid - hip_mid)), shoulder_mid - hip_mid) * np.sign(np.dot(keypoints['left_elbow'] - keypoints['left_shoulder'], D_s.T))[1]  # Not sure if this is right
    angles['shoulder_side_r'] = calculate_angle_vector(orthogonal_projection(keypoints['right_elbow'] - keypoints['right_shoulder'], np.cross(D_s, shoulder_mid - hip_mid)), shoulder_mid - hip_mid) * np.sign(np.dot(keypoints['right_elbow'] - keypoints['right_shoulder'], D_s.T))[1]  # Not sure if this is right
    angles['shoulder_abduc_l'] = calculate_angle_vector( keypoints['left_elbow'] - keypoints['left_shoulder'],keypoints['left_hip'] - keypoints['left_shoulder'])
    angles['shoulder_abduc_r'] = calculate_angle_vector( keypoints['right_elbow'] - keypoints['right_shoulder'],keypoints['right_hip'] - keypoints['right_shoulder'])
    angles['shoulder_flex_l'] = 90 - calculate_angle_vector( keypoints['left_elbow'] - keypoints['left_shoulder'], D_s)
    angles['shoulder_flex_r'] = 90 - calculate_angle_vector( keypoints['right_elbow'] - keypoints['right_shoulder'], D_s)
    angles['elbow_angle_l'] = calculate_angle_point(keypoints['left_shoulder'], keypoints['left_elbow'], keypoints['left_wrist'])
    angles['elbow_angle_r'] = calculate_angle_point(keypoints['right_shoulder'], keypoints['right_elbow'], keypoints['right_wrist'])

    return angles


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
            line=dict(color=colors[i % len(colors)], width=6)
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


def plot_box(angle_error):
    """
    Plot different errors in a box plot.
    :param errors_dict: Dictionary where keys are metric names and values are lists of error values.
    """
    fig = go.Figure()

    # Different colors and symbols for different metrics
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'black', 'pink']
    symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right', 'pentagon', 'hexagon']

    for i, (metric, errors) in enumerate(angle_error.items()):
        fig.add_trace(go.Box(y=errors, name=metric, boxmean=True, marker_color=colors[i % len(colors)], marker_symbol=symbols[i % len(symbols)]))

    fig.update_layout(
        title='Error Metrics',
        xaxis_title='Metrics',
        yaxis_title='Error Values',
        showlegend=True
    )

    wandb.log({"Angular error box": fig})



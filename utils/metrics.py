import numpy as np
from scipy import signal, stats
from sklearn.metrics import auc


def align_by_pelvis(joints):
    """
    Align the input joints by the pelvis joint
    :param joints: 3D joint positions
    :return: Aligned 3D joint positions
    """

    pelvis = joints[:, 0, :]
    joints = joints - pelvis
    return joints


def calculate_mpjpe(target, prediction):
    """
    Mean per-joint position error (MPJPE)
    :param target: Ground truth 3D joint positions
    :param prediction: Predicted 3D joint positions
    :return: Mean and standard deviation of the MPJPE
    """
    assert prediction.shape == target.shape

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
            assert (gt.shape[1] == pred.shape[1])

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
    Procrustes MJPE: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    :param target: Ground truth 3D joint positions
    :param prediction: Predicted 3D joint positions
    :return: Mean and standard deviation of the PMPJPE
    """
    assert prediction.shape == target.shape

    target, prediction, error_count = align_procrustes(target, prediction)
    mean, std = calculate_mpjpe(target, prediction)

    return mean, std, error_count


def calculate_pck(target, prediction, threshold=100.0,
                  joints_to_use=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], procrustes=False):
    """
    Calculate percentage of correct keypoints (PCK) in [%] (https://arxiv.org/pdf/1611.09813.pdf)
    :param target: Ground truth 3D joint positions
    :param prediction: Predicted 3D joint positions
    :param threshold: Threshold for correct keypoints
    :param joints_to_use: List of joints to use for PCK calculation
    :param procrustes: Whether to use procrustes alignment
    :return: PCK in [%]
    """

    assert len(prediction.shape) == len(target.shape)

    if procrustes:
        target, prediction, err = align_procrustes(target, prediction)

    distance = np.linalg.norm(target - prediction, axis=2)

    pck = distance <= threshold
    pck = np.mean(pck[:, joints_to_use], axis=1)
    pck = np.mean(pck) * 100.

    return pck


def mean_velocity_error(prediction, target, procrustes=True):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    :param prediction: Predicted 3D joint positions
    :param target: Ground truth 3D joint positions
    :param procrustes: Whether to use procrustes alignment
    :return: Mean and standard deviation of the mean per-joint velocity error
    """

    assert prediction.shape == target.shape

    velocity_predicted = np.diff(prediction, axis=0)
    velocity_target = np.diff(target, axis=0)

    if procrustes:
        mean, std, err = calculate_pmpjpe(velocity_target, velocity_predicted)
    else:
        mean, std, err = calculate_mpjpe(velocity_target, velocity_predicted)

    return mean, std


def mean_acceleration_error(prediction, target, procrustes=True):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    :param prediction: Predicted 3D joint positions
    :param target: Ground truth 3D joint positions
    :param procrustes: Whether to use procrustes alignment
    :return: Mean and standard deviation of the mean per-joint velocity error
    """

    assert prediction.shape == target.shape

    if procrustes:
        target, prediction, err = align_procrustes(target, prediction)


    acceleration_predicted = np.diff(np.diff(prediction, axis=0), axis=0)
    acceleration_target = np.diff(np.diff(target, axis=0), axis=0)

    mean, std, err = calculate_pmpjpe(acceleration_target, acceleration_predicted)

    return mean, std


def calculate_cmc(target, prediction, axes_to_use=[0, 1, 2], procrustes=False):
    """
    Calculate coefficient of multiple correlation (CMC) for all joints and axes
    :param target: Ground truth 3D joint positions, shape [sample, joint, 3]
    :param prediction: Predicted 3D joint positions, shape [sample, joint, 3]
    :param joints_to_use: List of joints to use for CMC calculation
    :param axes_to_use: List of axes to use for CMC calculation
    :param procrustes: Whether to use procrustes alignment
    :return: CMC
    :return: P-value
    """

    assert len(prediction) == len(target)

    target = target[:, :, axes_to_use]
    prediction = prediction[:, :, axes_to_use]

    if procrustes:
        target, prediction, err = align_procrustes(target, prediction)

    cmc_all = []
    pvalues_all = []
    for keypoint in range(target.shape[1]):
        for coordinates in range(target.shape[2]):
            gt = target[:, keypoint, coordinates]
            pred = prediction[:, keypoint, coordinates]
            gt_norm = np.linalg.norm(gt)
            pred_norm = np.linalg.norm(pred)
            gt = gt / gt_norm
            pred = pred / pred_norm
            cmc, pvalue = stats.pearsonr(gt, pred) # ValueError: array must not contain infs or NaNs
            cmc_all.append(cmc)
            pvalues_all.append(pvalue)

    return np.mean(cmc_all), np.mean(pvalues_all)


def compute_CP(target, prediction, threshold=180, joints_to_use=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]):
    """
    Compute the number of correct poses
    :param target: Ground truth 3D joint positions
    :param prediction: Predicted 3D joint positions
    :param threshold: Threshold for correct poses
    :param joints_to_use: List of joints to use for CP calculation
    :return: Number of correct poses
    """
    assert len(target.shape) == len(prediction.shape)

    distances = np.linalg.norm(target[:, joints_to_use, :] - prediction[:, joints_to_use, :], axis=2)
    correct_poses = np.count_nonzero(distances < threshold, axis=1) == len(joints_to_use)

    return correct_poses


def compute_CPS(target, prediction, min_th=1, max_th=300, step=1,
                joints_to_use=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], procrustes=False):
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

    assert len(prediction.shape) == len(target.shape)

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
    :return joint_angles: [frames, joint, 1]
    """

    all_angles = []
    for i in range(joint_a.shape[0]):
        a = joint_a[i]
        b = joint_b[i]
        c = joint_c[i]
        v1 = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
        v2 = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])

        v1mag = np.sqrt([v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]])
        v1norm = np.array([v1[0] / v1mag, v1[1] / v1mag, v1[2] / v1mag])

        v2mag = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
        v2norm = np.array([v2[0] / v2mag, v2[1] / v2mag, v2[2] / v2mag])
        res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
        angle_rad = np.arccos(res)
        all_angles.append(angle_rad)
    all_angles = np.concatenate(np.array(all_angles))
    return np.round(np.degrees(all_angles), 2)


def angle_between_points(joint_a, joint_b, joint_c):
    """
    Calculate the angle between two segments in 3D
    :param joint_a: [frames, joint, 3]
    :param joint_b: [frames, joint, 3]
    :param joint_c: [frames, joint, 3]
    :return joint_angles: [frames, joint, 1]
    """
    all_angles = []
    for i in range(joint_a.shape[0]):
        AB = np.array(joint_b[i]) - np.array(joint_a[i])
        BC = np.array(joint_c[i]) - np.array(joint_b[i])

        dot_product = np.dot(AB, BC)

        magnitude_AB = np.linalg.norm(AB)
        magnitude_BC = np.linalg.norm(BC)

        angle_rad = np.arccos(dot_product / (magnitude_AB * magnitude_BC))
        angle_deg = np.degrees(angle_rad)
        all_angles.append(angle_deg)
    all_angles = np.concatenate(np.array(all_angles))
    return all_angles


def orthogonal_projection(A, B, n_c, n_d):
    """
    Calculate the orthogonal projection of vector AB onto the plane defined by the normal vector n
    :param A:
    :param B:
    :param n_c:
    :param n_d:
    :return:
    """
    all_proj = []
    for i in range(A.shape[0]):
        AB = np.array(B[i]) - np.array(A[i])
        n = np.array(n_c[i]) - np.array(n_d[i])
        proj = AB - np.dot(AB, n) / np.dot(n, n) * n
        all_proj.append(proj)
    all_proj = np.concatenate(np.array(all_proj))
    return all_proj


def help_shoulder(hip_mid, shoulder_mid, shoulder_left, shoulder_right):
    """
    Calculate the help vector Ds = (hip_mid, shoulder_mid) x (shoulder_right, shoulder_left)
    :param hip_center: Hip center
    :param shoulder_mid: Shoulder center
    :param shoulder_left: Left shoulder
    :param shoulder_right: Right shoulder
    :return: Help vector Ds
    """
    all_ds = []
    for i in range(hip_mid.shape[0]):
        _hip_mid = np.array(hip_mid[i])
        _shoulder_mid = np.array(shoulder_mid[i])
        _shoulder_left = np.array(shoulder_left[i])
        _shoulder_right = np.array(shoulder_right[i])

        ds = np.cross(_hip_mid - _shoulder_mid, _shoulder_right - _shoulder_left)
        all_ds.append(ds)
    all_ds = np.concatenate(np.array(all_ds))
    return all_ds


def help_hip(Y, hip_left, hip_right):
    """
    Calculate the help vector Dh = Y x (hip_right, hip_left)
    :param Y: Y vector
    :param hip_left: Left hip
    :param hip_right: Right hip
    :return: Help vector Dh
    """
    all_dh = []
    for i in range(Y.shape[0]):
        _Y = np.array(Y[i])
        _hip_left = np.array(hip_left[i])
        _hip_right = np.array(hip_right[i])

        dh = np.cross(_Y, _hip_right - _hip_left)
        all_dh.append(dh)
    all_dh = np.concatenate(np.array(all_dh))
    return all_dh


def shoulder_mid(shoulder_left, shoulder_right):
    """
    Calculate the shoulder mid point
    :param shoulder_left: Left shoulder
    :param shoulder_right: Right shoulder
    :return: Shoulder mid point
    """
    all_shoulder_mid = []
    for i in range(shoulder_right.shape[0]):
        _shoulder_left = np.array(shoulder_left[i])
        _shoulder_right = np.array(shoulder_right[i])

        shoulder_mid = (_shoulder_right + _shoulder_left) / 2
        all_shoulder_mid.append(shoulder_mid)
    all_shoulder_mid = np.concatenate(np.array(all_shoulder_mid))
    return all_shoulder_mid


def hip_mid(hip_left, hip_right):
    """
    Calculate the hip mid point
    :param hip_left: Left hip
    :param hip_right: Right hip
    :return: Hip mid point
    """
    all_hip_mid = []
    for i in range(hip_left.shape[0]):
        _hip_left = np.array(hip_left[i])
        _hip_right = np.array(hip_right[i])

        hip_mid = (_hip_right + _hip_left) / 2
        all_hip_mid.append(hip_mid)
    all_hip_mid = np.concatenate(np.array(all_hip_mid))
    return all_hip_mid


def trunk_angle(hip_mid, shoulder_mid, Ds):
    """
    Calculate the trunk angle
    :param hip_mid: Hip mid point
    :param shoulder_mid: Shoulder mid point
    :return: Trunk angle
    """
    all_trunk_angle = []
    for i in range(hip_mid.shape[0]):
        _hip_mid = np.array(hip_mid[i])
        _shoulder_mid = np.array(shoulder_mid[i])
        _Ds = np.array(Ds[i])

        trunk_angle = 90 - angle_between_points(_hip_mid, _shoulder_mid, _Ds)
        all_trunk_angle.append(trunk_angle)
    all_trunk_angle = np.concatenate(np.array(all_trunk_angle))
    return all_trunk_angle


def trunk_twist(hip_left, hip_right, shoulder_left, shoulder_right):
    """
    Calculate the trunk twist
    :param hip_left: Left hip
    :param hip_right: Right hip
    :param shoulder_left: Left shoulder
    :param shoulder_right: Right shoulder
    :return: Trunk twist
    """
    all_trunk_twist = []
    for i in range(hip_mid.shape[0]):
        _hip_left = np.array(hip_left[i])
        _hip_right = np.array(hip_right[i])
        _shoulder_left = np.array(shoulder_left[i])
        _shoulder_right = np.array(shoulder_right[i])

        _shoulder_mid = shoulder_mid(_shoulder_left, _shoulder_right)
        _hip_mid = hip_mid(_hip_left, _hip_right)

        trunk_twist = angle_between_points(orthogonal_projection(_hip_left, _hip_right, _hip_mid, _shoulder_mid),
                                           orthogonal_projection(_shoulder_left, _shoulder_right, _hip_mid, _shoulder_mid))

        all_trunk_twist.append(trunk_twist)
    all_trunk_twist = np.concatenate(np.array(all_trunk_twist))
    return all_trunk_twist


def trunk_bending(Y, hip_mid, hip_left, hip_right):
    """
    Calculate the trunk bending
    :param Y: Y vector
    :param hip_mid: Hip mid point
    :param hip_left: Left hip
    :return: Trunk bending
    """
    all_trunk_bending = []
    for i in range(Y.shape[0]):
        _Y = np.array(Y[i])
        _hip_mid = np.array(hip_mid[i])
        _shoulder_mid = np.array(shoulder_mid[i])
        _helper_hip = np.array(help_hip(_Y, hip_left[i], hip_right[i]))

        trunk_bending = angle_between_points(_Y, orthogonal_projection(_hip_mid, _shoulder_mid, _helper_hip))
        all_trunk_bending.append(trunk_bending)
    all_trunk_bending = np.concatenate(np.array(all_trunk_bending))
    return all_trunk_bending


def knee_angle(hip_left, hip_knee, knee_left):
    """
    Calculate the knee angle
    :param hip_left: Left hip
    :param hip_knee: Left knee
    :param knee_ankle: Left ankle
    :return: Knee angle
    """
    all_knee_angle = []
    for i in range(hip_left.shape[0]):
        _hip_left = np.array(hip_left[i])
        _hip_knee = np.array(hip_knee[i])
        _knee_ankle = np.array(knee_left[i])

        knee_angle = angle_between_points(_hip_left, _hip_knee, _knee_ankle)
        all_knee_angle.append(knee_angle)
    all_knee_angle = np.concatenate(np.array(all_knee_angle))
    return all_knee_angle

####################################################################
### Adjust according to https://www.mdpi.com/1424-8220/22/5/1729 ###
####################################################################
def calculate_mpsae(target, prediction, joint_segments):
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

    # Help Vector Ds = (hip_mid, shoulder_mid) x (shoulder_right, shoulder_left)
    # Help Vector Dh = Y x (hip_right, hip_left)
    # Trunk angle: 90degree - angle((hip_mid, shoulder_mid), Dh)
    # Trunk twist: angle(project((hip_left, hip_right), (hip_mid, shoulder_mid), project((shouler_left, shoulder_right), (hip_mid, shoulder_mid)))
    # Trunk bending: angle(Y, project((hip_mid, hip_left), Dh))
    # Knee angle: angle(hip_left, hip_k, left_ankle)
    # Shoulder angle: angle(project(((ellbow_left, shoulder_left), Ds x (hip_mid, shoulder_mid)), (hip_mid, shoulder_mid)) * sgn((shoulder_left, ellbow_left, Ds)
    # Elbow angle: angle(shoulder_left, ellbow_left, wrist_left)
    # Ankle angle:

    return np.mean(single_segment_err), np.std(single_segment_err), segment_errors_mean, segment_errors_std


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

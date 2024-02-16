import numpy as np
from scipy import signal, stats
from sklearn.metrics import auc, r2_score


def align_by_pelvis(joints):
    """
    Align the input joints by the pelvis joint
    :param joints: 3D joint positions
    :return: Aligned 3D joint positions
    """

    pelvis = joints[:, 0, :]
    joints = joints - pelvis
    return joints


def calculate_mpjpe(target: object, prediction: object) -> object:
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


def calculate_pmpjpe(target, prediction):
    """
    Procrustes MJPE: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    :param target: Ground truth 3D joint positions
    :param prediction: Predicted 3D joint positions
    :return: Mean and standard deviation of the PMPJPE
    """
    assert prediction.shape == target.shape

    pred_hat_all = []
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
            U, s, Vh = np.linalg.svd(K)
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
            pred_hat = np.tile(np.mean(gt, axis=0), (17, 1))
            R = np.identity(3)

        pa_error = mpjpe = np.linalg.norm(gt_raw - pred_hat, axis=1)
        pred_hat_all.append(np.mean(pa_error))

    return np.mean(pred_hat_all), np.std(pred_hat_all)


def calculate_pck(target, prediction, threshold=100.0, joints_to_use=[1, 2, 3, 4, 5, 6, 8, 10, 11]):
    """
    Calculate percentage of correct keypoints (PCK) in [%] (https://arxiv.org/pdf/1611.09813.pdf)
    :param target: Ground truth 3D joint positions
    :param prediction: Predicted 3D joint positions
    :param threshold: Threshold for correct keypoints
    :param joints_to_use: List of joints to use for PCK calculation
    :return: PCK in [%]
    """

    assert len(prediction.shape) == len(target.shape)

    distance = np.linalg.norm(target - prediction, axis=2)

    pck = distance <= threshold
    pck = np.mean(pck[:, joints_to_use], axis=1)
    pck = np.mean(pck) * 100.

    return pck


def mean_velocity_error(prediction, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    :param prediction: Predicted 3D joint positions
    :param target: Ground truth 3D joint positions
    :return: Mean and standard deviation of the mean per-joint velocity error
    """

    assert prediction.shape == target.shape

    velocity_predicted = np.diff(prediction, axis=0)
    velocity_target = np.diff(target, axis=0)

    mean, std = calculate_pmpjpe(velocity_target, velocity_predicted)

    return mean, std


def mean_acceleration_error(prediction, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    :param prediction: Predicted 3D joint positions
    :param target: Ground truth 3D joint positions
    :return: Mean and standard deviation of the mean per-joint velocity error
    """

    assert prediction.shape == target.shape

    acceleration_predicted = np.diff(np.diff(prediction, axis=0), axis=0)
    acceleration_target = np.diff(np.diff(target, axis=0), axis=0)

    mean, std = calculate_pmpjpe(acceleration_target, acceleration_predicted)

    return mean, std


def calculate_cmc(target, prediction, joints_to_use=[1, 2, 3, 4, 5, 6, 8, 10, 11], axes_to_use=[0, 1, 2]):
    """
    Calculate coefficient of multiple correlation (CMC) for all joints and axes
    :param target: Ground truth 3D joint positions, shape [sample, joint, 3]
    :param prediction: Predicted 3D joint positions, shape [sample, joint, 3]
    :param joints_to_use: List of joints to use for CMC calculation
    :param axes_to_use: List of axes to use for CMC calculation
    :return: CMC
    :return: P-value
    """

    assert len(prediction) == len(target)

    target = target[:, joints_to_use, :][:, :, axes_to_use]
    prediction = prediction[:, joints_to_use, :][:, :, axes_to_use]

    cmc_all = []
    pvalues_all = []
    for keypoints in range(target.shape[1]):
        for coordinates in range(target.shape[2]):
            gt = target[:, keypoints, coordinates]
            pred = prediction[:, keypoints, coordinates]
            cmc, pvalue = stats.pearsonr(gt, pred)
            cmc_all.append(cmc)
            pvalues_all.append(pvalue)

    return np.mean(cmc_all), np.mean(pvalues_all)


def compute_CP(target, prediction, threshold=180, joints_to_use=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]):
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


def compute_CPS(target, prediction, min_th=1, max_th=300, step=1, joints_to_use=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]):
    """
    Compute the correct pose score (CPS) according to (https://arxiv.org/abs/2011.14679) for different thresholds
    :param target: Ground truth 3D joint positions, shape [sample, joint, 3]
    :param prediction: Predicted 3D joint positions, shape [sample, joint, 3]
    :param min_th: Minimum threshold
    :param max_th: Maximum threshold
    :param step: Step size
    :param joints_to_use: List of joints to use for CPS calculation
    :return: CPS
    :return: idx of best CPS
    """

    assert len(prediction.shape) == len(target.shape)

    cps_length = int((max_th + 1 - min_th) / step)
    thresholds = np.arange(min_th, max_th + 1, step)
    cps_best_list = np.zeros(cps_length, dtype=np.double)
    cp_values_list = np.empty((prediction.shape[0], len(thresholds)), dtype=np.double)

    for i, threshold in enumerate(thresholds):
        cp_values_list[:, i] = compute_CP(target, prediction, threshold, joints_to_use)

    values = np.max(cp_values_list, axis=1)
    cps_idx = np.argmax(cp_values_list, axis=1)
    cps_best_list += np.sum(values)

    cps_best_list /= prediction.shape[0]
    cps_best = auc(thresholds, cps_best_list)

    return cps_best, cps_idx


def angle_2p_3d(joint_a, joint_b, joint_c):
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


## NOT WORKING
def calculate_mpsae(target, prediction, segments):
    """
    Calculate the mean per segment angle error. Target and prediction are dictionaries with the same keys
    [batch, joint, 3] segments is a list of lists of segment names
    [[child1a, parent1, child1c], [child2a, parent2, child2c], ...]
    """

    assert len(prediction.shape) == len(target.shape)

    angle_errors = []
    for i, segment in enumerate(segments):
        angle_target = angle_2p_3d(target[:, segment[0]], target[:, segment[1]], target[:, segment[2]])  # batch, 1
        angle_prediction = angle_2p_3d(prediction[:, segment[0]], prediction[:, segment[1]], prediction[:, segment[2]])
        angle_err = np.abs(np.array(angle_target) - np.array(angle_prediction))
        angle_err = np.round(angle_err, 2)
        angle_errors.append(angle_err)  # segments, batch, 1
    single_segment_err = np.mean(np.array(angle_errors), axis=1)
    single_segment_std = np.std(np.array(angle_errors), axis=1)

    return np.mean(single_segment_err), np.std(single_segment_err), single_segment_err, single_segment_std


def calculate_angle(joint0, joint1, joint2):
    """
    Calculate angle between two segments
    """
    v1 = joint0 - joint1
    v2 = joint2 - joint1
    return np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))


def calculate_rom(target, prediction, segments):
    """
    Calculate range of motion (ROM)
    """

    assert prediction.shape == target.shape

    for segment in enumerate(segments):
        target_angle = calculate_angle(target[i, segment[0], :], target[i, segment[1], :], target[i, segment[2], :])
        prediction_angle = calculate_angle(prediction[i, segment[0], :], prediction[i, segment[1], :],
                                           prediction[:, segment[2], :])

    target_ROM = np.max(target_angle) - np.min(target_angle)
    prediction_ROM = np.max(prediction_angle) - np.min(prediction_angle)

    max = np.sum(np.abs(np.max(prediction_angle) - np.max(target_angle)) / len(target_angle), axis=0)
    min = np.sum(np.abs(np.min(prediction_angle) - np.min(target_angle)) / len(target_angle), axis=0)
    ROM = np.sum(np.abs(prediction_ROM - target_ROM) / len(target_ROM), axis=0)

    signed_max = np.sum(np.max(prediction_angle) - np.max(target_angle) / len(target_angle), axis=0)
    signed_min = np.sum(np.min(prediction_angle) - np.min(target_angle) / len(target_angle), axis=0)
    signed_ROM = np.sum(prediction_ROM - target_ROM / len(target_ROM), axis=0)

    return signed_ROM, signed_max, signed_min, ROM, max, min


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


def calculate_symmetry_error(poses, reduction='none', dim=-1):
    bones = {
        'h36m': torch.tensor([[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [7, 10], [10, 11],
                           [11, 12], [7, 13], [13, 14], [14, 15]], device=poses.device),
        '3dpw': torch.tensor([[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9],
                      [7, 10], [8, 11], [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], [14, 17],
                      [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]], device=poses.device)
             }

    # TODO: Fix support for SIMPL joints

    bone_pairs = {'h36m': torch.tensor([[0, 2], [1, 3], [7, 10], [8, 11], [9, 12]], device=poses.device)}
    bone_indices = bones["h36m"] + 1

    start_pos = torch.index_select(poses, -1, bone_indices[:, 0])
    end_pos = torch.index_select(poses, -1, bone_indices[:, 1])

    # Extract bones as delta positions and calculate length
    bone_lengths = torch.linalg.norm(end_pos - start_pos, dim=-2)
    # Find matching bones in skeleton
    bone0 = torch.index_select(bone_lengths, -1, bone_pairs['h36m'][:, 0])
    bone1 = torch.index_select(bone_lengths, -1, bone_pairs['h36m'][:, 1])

    # Calculate the absolute length difference between symmetries
    absolute_error = torch.abs(bone0 - bone1)

    # Calculate the average error for all bones
    absolute_error = absolute_error.mean(dim=-1)

    if reduction == 'none':
        return absolute_error
    elif reduction == 'mean':
        return absolute_error.mean(dim=dim)
    elif reduction == 'sum':
        return absolute_error.sum(dim=dim)
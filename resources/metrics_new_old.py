import numpy as np
from scipy import signal
from sklearn.metrics import auc, r2_score


def calculate_mpjpe(target: object, prediction: object) -> object:
    """
    Doesn't work in combination with skeletal morphing, as the outputs are not scaled correctly.
    input = (batch_size, num_joints, 3)
    """
    assert prediction.shape == target.shape
    mean = np.mean(np.linalg.norm(prediction - target, axis=len(target.shape) - 1))
    std = np.std(np.linalg.norm(prediction - target, axis=len(target.shape) - 1))

    return mean, std


def calculate_pmpjpe(target, prediction):
    """
    Procrustes MJPE: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert prediction.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(prediction, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = prediction - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)

    try:
        U, _, V = np.linalg.svd(H)
    except:
        return np.nan, np.nan

    u, s, v_t = np.linalg.svd(H)
    v = v_t.transpose(0, 2, 1)
    r = np.matmul(v, u.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_det_r = np.sign(np.expand_dims(np.linalg.det(r), axis=1))
    v[:, :, -1] *= sign_det_r
    s[:, -1] *= sign_det_r.flatten()
    r = np.matmul(v, u.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, r)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(prediction, r) + t

    # Return MPJPE
    mean = np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))
    std = np.std(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))

    return mean, std


def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)

    mean = np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape) - 1))
    std = np.std(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape) - 1))

    return mean, std


def mean_acceleration_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)

    acceleration_predicted = np.diff(velocity_predicted, axis=0)
    acceleration_target = np.diff(velocity_target, axis=0)

    mean = np.mean(np.linalg.norm(np.abs(acceleration_target - acceleration_predicted), axis=len(target.shape) - 1))
    std = np.std(np.linalg.norm(np.abs(acceleration_target - acceleration_predicted), axis=len(target.shape) - 1))

    return mean, std


def compute_CP(target, prediction, threshold=180):
    """
    Compute the number of correct poses
    """
    assert len(target.shape) == len(prediction.shape)

    joints_to_use = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    distances = np.sqrt(np.sum((target[:, joints_to_use, :]
                                - prediction[:, joints_to_use, :]) ** 2,
                               axis=1))  # maybe use procustes distance instead of MPJPE
    correct_poses = np.count_nonzero(distances < threshold, axis=1) == len(joints_to_use)
    return correct_poses

## NOT WORKING
def compute_CPS(target, prediction, min_th=1, max_th=300, step=1):
    """
    Compute the correct pose score (CPS) according to (https://arxiv.org/abs/2011.14679) for different thresholds
    """

    assert len(prediction.shape) == len(target.shape)

    thresholds = np.arange(min_th, max_th + 1, step)
    cp_values_list = np.empty((prediction.shape[0], len(thresholds)), dtype=np.double)
    for i, threshold in enumerate(thresholds):
        cp_values_list[:, i] = compute_CP(target, prediction, threshold=threshold)

    values, _ = np.max(cp_values_list, axis=0)
    cps_best_list = np.sum(values, axis=0)

    values, _ = np.min(cp_values_list, axis=0)
    cps_worst_list = np.sum(values, axis=0)

    k_list = np.arange(min_th, max_th + 1, step)
    cps_best_list /= prediction.size(0)
    cps_best = auc(k_list, cps_best_list)

    cps_worst_list /= prediction.size(0)
    cps_worst = auc(k_list, cps_worst_list)

    return cps_best, cps_worst


## NOT WORKING
def calculate_cmc(target, prediction):
    """
    Calculate coefficient of multiple correlation (CMC) for all joints and axes
    """

    assert len(prediction) == len(target)

    # normalize inputs (or calculate euclidian norm first?)
    norm_predicted = np.mean(np.sum(prediction ** 2, axis=2, keepdims=True), axis=1, keepdims=True)
    norm_target = np.mean(np.sum(target * prediction, axis=2, keepdims=True), axis=1, keepdims=True)
    scale = norm_target / norm_predicted
    norm_predicted = prediction * scale

    # calculate CMC based on normalized inputs per joint and axis
    cmc = []
    for i in range(target.shape[1]):
        for j in range(target.shape[2]):
            r2 = r2_score(target[:, i, j], norm_predicted[:, i, j])
            _cmc = np.square(r2)
            cmc.append(_cmc)
    cmc = np.mean(cmc)
    return cmc


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


## NOT WORKING
def calculate_pck(target, prediction, threshold=150, joints_to_use = [1, 2, 3, 4, 5, 6, 8, 10, 11]):
    """ Calculate percentage of correct keypoints (PCK) in [%]"""
    assert len(prediction.shape) == len(target.shape)

    # see https://arxiv.org/pdf/1611.09813.pdf
    distances = np.sqrt(np.sum((target[:, joints_to_use, :]
                                - prediction[:, joints_to_use, :]) ** 2, axis=1))
    pck = np.count_nonzero(distances < threshold, axis=1) / len(joints_to_use)
    return pck


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

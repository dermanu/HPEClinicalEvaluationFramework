import numpy as np
from sklearn.metrics import auc


def calculate_mpjpe(target, prediction):
    """ Doesn't work in combination with skeletal morphing, as the outputs are not scaled correctly.
         input = (batch_size, num_joints, 3)"""
    assert prediction.shape == target.shape
    mean = np.mean(np.linalg.norm(prediction - target, axis=len(target.shape) - 1))
    std = np.std(np.linalg.norm(prediction - target, axis=len(target.shape) - 1))

    return mean, std


def calculate_weighted_mpjpe(target, prediction, weights):
    """ Calculate weighted mean per joint position error (MPJPE) """
    assert prediction.shape == target.shape
    assert weights.shape[0] == prediction.shape[0]

    mean = np.mean(weights * np.linalg.norm(prediction - target, axis=len(target.shape) - 1))
    std = np.std(weights * np.linalg.norm(prediction - target, axis=len(target.shape) - 1))

    return mean, std


def calculate_nmpjpe(target, prediction):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert prediction.shape == target.shape

    norm_predicted = np.mean(np.sum(prediction ** 2, axis=2, keepdims=True), axis=1, keepdims=True)
    norm_target = np.mean(np.sum(target * prediction, axis=2, keepdims=True), axis=1, keepdims=True)
    scale = norm_target / norm_predicted

    mean, std = calculate_mpjpe(scale * prediction, target)

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
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(prediction, R) + t

    # Return MPJPE
    mean = np.mean(np.linalg.norm(predicted_aligned - target, ord='fro', axis=len(target.shape) - 1))
    std = np.std(np.linalg.norm(predicted_aligned - target, ord='fro', axis=len(target.shape) - 1))

    return mean, std


def calculate_pcp(target, prediction, keypoints_target, keypoints_prediction, parts):
    """ Calculate percentage of correct part (PCP) """


def calculate_pck(target, prediction, threshold=150, joints_to_use=[1, 2, 3, 4, 5, 6, 8, 10, 11]):
    """ Calculate percentage of correct keypoints (PCK) in [%]"""
    assert len(prediction.shape) == len(target.shape)

    # see https://arxiv.org/pdf/1611.09813.pdf
    distances = np.sqrt(np.sum((target[:, joints_to_use, :]
                                - prediction[:, joints_to_use, :]) ** 2, axis=1))
    pck = np.count_nonzero(distances < threshold, axis=1) / len(joints_to_use)
    return pck


def calculate_cp(target, prediction, threshold=180):
    """ Calculate correct pose (CP) based on Protocol 2 """
    assert prediction.shape == target.shape

    err = []
    for sample in range(prediction.shape[0]):
        pmpjpe = []
        for keypoint in range(prediction.shape[1]):
            pmpjpe.append(calculate_pmpjpe(target[sample, keypoint, :], prediction[sample, keypoint, :]))
        if any(pmpjpe) > threshold:
            err.append(0)
        else:
            err.append(1)
    return sum(err) / prediction.shape[0] * 100  # percentage of correct poses


def compute_CP(target, prediction, threshold=180):
    # maybe use procustes distance instead of MPJPE
    assert len(target.shape) == len(prediction.shape)

    joints_to_use = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    distances = np.sqrt(np.sum((target[:, joints_to_use, :]
                                - prediction[:, joints_to_use, :]) ** 2, axis=1))
    correct_poses = np.count_nonzero(distances < threshold, axis=1) == len(joints_to_use)
    return correct_poses


def compute_CPS(target, prediction, min_th=1, max_th=300, step=1):
    # computes Correct Poses Score (CPS) (https://arxiv.org/abs/2011.14679)
    # for different thresholds
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


def calculate_angle(joint0, joint1, joint2):
    """ Calculate angle between two segments """
    v1 = joint0 - joint1
    v2 = joint2 - joint1
    return np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))


def calculate_angle_error(target, prediction, segments):
    """ Calculate joint angle error (euler angle) """
    assert prediction.shape == target.shape

    angles = []
    for segment in enumerate(segments):
        for i in range(prediction.shape[0]):
            angle_diff = calculate_angle(target[i, segment[0], :], target[i, segment[1], :], target[i, segment[2], :])\
                        - calculate_angle(prediction[i, segment[0], :], prediction[i, segment[1], :], prediction[:, segment[2], :])
            angle_diff = np.abs(angle_diff)


    angles = np.array(angles)

    rmse = np.sqrt(np.mean(angles_diff ** 2))

    return np.mean(angles), np.std(angles), rmse


def calculate_rom(target, prediction, segments):
    """ Calculate range of motion (ROM) """

    assert prediction.shape == target.shape

    for segment in enumerate(segments):
        target_angle = calculate_angle(target[i, segment[0], :], target[i, segment[1], :], target[i, segment[2], :])
        prediction_angle = calculate_angle(prediction[i, segment[0], :], prediction[i, segment[1], :], prediction[:, segment[2], :])

    target_ROM = np.max(target_angle) - np.min(target_angle)
    prediction_ROM = np.max(prediction_angle) - np.min(prediction_angle)

    max = np.sum(np.abs(np.max(prediction_angle) - np.max(target_angle)) / len(target_angle), axis=0)
    min = np.sum(np.abs(np.min(prediction_angle) - np.min(target_angle)) / len(target_angle), axis=0)
    ROM = np.sum(np.abs(prediction_ROM - target_ROM) / len(target_ROM), axis=0)

    signed_max = np.sum(np.max(prediction_angle) - np.max(target_angle) / len(target_angle), axis=0)
    signed_min = np.sum(np.min(prediction_angle) - np.min(target_angle) / len(target_angle), axis=0)
    signed_ROM = np.sum(prediction_ROM - target_ROM / len(target_ROM), axis=0)

    return signed_ROM, signed_max, signed_min, ROM, max, min

def calculate_cmc(target, prediction):
    calculate_angle
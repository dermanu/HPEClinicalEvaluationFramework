import numpy as np
import scipy.signal
from sklearn.metrics import auc, r2_score
from scipy import signal


###################
# Calculate MPJPE #
###################

def calculate_mpjpe(target, prediction):
    """ Doesn't work in combination with skeletal morphing, as the outputs are not scaled correctly.
         input = (batch_size, num_joints, 3)"""
    assert prediction.shape == target.shape
    mean = np.mean(np.linalg.norm(prediction - target, axis=len(target.shape) - 1))
    std = np.std(np.linalg.norm(prediction - target, axis=len(target.shape) - 1))

    return mean, std


####################
# Calculate NMPJPE #
####################

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


###################
# Calculate MPJVE #
###################

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


###################
# Calculate MPJAE #
###################

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


####################
# Calculate PMPJPE #
####################

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
    mean = np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))
    std = np.std(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))

    return mean, std


####################
# Calculate PCK3D #
####################

# https://github.com/hrussel/t-leap


def calc_dists(target, prediction, ):
    dists = np.zeros((prediction.shape[0], prediction.shape[1]))
    for b in range(prediction.shape[0]):
        for k in range(prediction.shape[1]):
            if target[b, k, 0] > 1 and target[b, k, 1] > 1:
                dists[b, k] = np.linalg.norm(prediction[b, k, :] - target[b, k, :])
            else:
                dists[b, k] = -1
    return dists


def dist_acc(dists, thr):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.equal(dists, -1)
    num_dist_cal = np.sum(dist_cal)
    if num_dist_cal > 0:
        return np.sum(np.less(dists[dist_cal], thr)) * 1.0 / num_dist_cal
    else:
        return -1


def calculate_pck(target, prediction, thr=150):
    """
    Calculates the percentage of correct keypoints
    :param targets: the target keypoints
    :param predictions: the predicted HEATMAPS
    :param thr: threshold under which a keypoint is considered correct
    :return: the PCK per batch image, the mean PCK, and the number of correct keypoints per batch image
    """
    return _accuracy(target, prediction, thr)


def _accuracy(target, prediction, thr):
    """
        Calculates the percentage of correct keypoints
        :param targets: the target keypoints
        :param predictions: the predicted HEATMAPS
        :param thr: threshold under which a keypoint is considered correct
        :return: the PCK per batch image, the mean PCK, and the number of correct keypoints per batch image
        """
    dists = calc_dists(target, prediction)

    batch_size = target.shape[0]
    n_keypoints = target.shape[1]

    acc = np.zeros((batch_size, n_keypoints))
    cnt = [0] * batch_size

    for b in range(batch_size):
        for k in range(n_keypoints):
            acc_b_k = dist_acc(dists[b, k], thr)
            if acc_b_k > 0:
                acc[b, k] = acc_b_k
                cnt[b] += 1

    return np.mean(acc)

def calculate_3dpck(target, prediction, threshold=150):
    # poses_pred.shape (bs, 3, 17)
    assert prediction.shape == target.shape
    # see https://arxiv.org/pdf/1611.09813.pdf

    distances = np.sqrt(np.sum((target - prediction)**2, axis=2))
    pck = np.count_nonzero(distances < threshold, axis=1) / prediction.shape[1]
    return np.sum(pck) / len(pck) * 100


#################
# Calculate CPS #
#################

def compute_CP(target, prediction, threshold=180):
    # maybe use Procustes distance instead of MPJPE
    assert len(target.shape) == len(prediction.shape)

    distances = np.sqrt(np.sum((target - prediction) ** 2, axis=2))
    correct_poses = np.count_nonzero(distances < threshold, axis=1) == prediction.shape[1]
    return correct_poses


def compute_CPS(target, prediction, min_th=1, max_th=300, step=1):
    # computes Correct Poses Score (CPS) (https://arxiv.org/abs/2011.14679)
    # https://github.com/twehrbein/Probabilistic-Monocular-3D-Human-Pose-Estimation-with-Normalizing-Flows/
    # for different thresholds

    assert len(prediction.shape) == len(target.shape)

    cp_list_length = (max_th + 1 - min_th) / step
    cps_best_list = np.zeros(np.int_(cp_list_length, ))

    thresholds = np.arange(min_th, max_th + 1, step)

    cp_values_list = np.empty((prediction.shape[0], len(thresholds)), dtype=np.double)
    for i, threshold in enumerate(thresholds):
        cp_values_list[:, i] = compute_CP(target, prediction, threshold=threshold)

    cp_values_list = np.array(cp_values_list, dtype=np.double)

    values = np.amax(cp_values_list, axis=0)
    cps_best_list += np.sum(values, axis=0)

    cps_best_list /= len(prediction)
    cps_best = auc(thresholds, cps_best_list)

    return cps_best


###################
# Calculate MPSAE #
###################

def angle_2p_3d(joint_a, joint_b, joint_c):
    all = []
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
        all.append(angle_rad)
    all = np.concatenate(np.array(all))
    return np.round(np.degrees(all), 2)


def calculate_mpsae(target, prediction, segments):
    # calculate the mean per segment angle error
    # target and prediction are dictionaries with the same keys [batch, joint, 3]
    # segments is a list of lists of segment names [[child1a, parent1, child1c], [child2a, parent2, child2c], ...]

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


####################
# Calculate MPJPhE #
####################

def wrapToPi(x):
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

    ## padding to remove edge effects (DOI: 10.1123/jab.2018-0396)
    pad_width = np.sqrt(target.shape[0]).astype(int)+1
    pad_width = ((pad_width**2-target.shape[0])/2).astype(int)
    target = np.pad(target, ((pad_width, pad_width), (0, 0), (0, 0)), 'constant', constant_values=0)
    prediction = np.pad(prediction, ((pad_width, pad_width), (0, 0), (0, 0)), 'constant', constant_values=0)

    ## calculate Hilbert transform
    h_target = signal.hilbert(target, axis=0)
    h_prediction = signal.hilbert(prediction, axis=0)

    ## calculate phase
    phase_target = np.unwrap(np.angle(h_target))
    phase_prediction = np.unwrap(np.angle(h_prediction))

    ## calculate phase difference
    phase_diff = wrapToPi(phase_target - phase_prediction)

    ## remove padding
    phase_diff = phase_diff[pad_width:-pad_width]

    return np.median(phase_diff), np.std(phase_diff), phase_target, phase_prediction, phase_diff

#################
# Calculate CMC #
#################
def calculate_cmc(target, prediction):
    # Calculate coefficient of multiple correlation (CMC) for all joints and axes
    # Maybe calculate the Euclidian norm first and then the CMC from there? Might be more important for CKC, but less
    # representative for the actual error
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
            r2 = r2_score(target[:,i,j], norm_predicted[:,i,j])
            _cmc = np.square(r2)
            cmc.append(_cmc)
    cmc = np.mean(cmc)
    return cmc
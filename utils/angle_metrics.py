def calculate_angle(joint_a, joint_b, joint_c):
    v1 = joint_a - joint_b
    v2 = joint_c - joint_b
    v1_norm = v1 / np.linalg.norm(v1, axis=2, keepdims=True)
    v2_norm = v2 / np.linalg.norm(v2, axis=2, keepdims=True)
    dot_product = np.sum(v1_norm * v2_norm, axis=2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
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


def mid_vector(shoulder_right, shoulder_left, hip_left, hip_right):
    """
    Calculate the shoulder and mid-point
    :param shoulder_left: Left shoulder
    :param shoulder_right: Right shoulder
    :return: Shoulder mid-point
    """
    shoulder_mid_point = (shoulder_right + shoulder_left) / 2
    hip_mid_point = (hip_right + hip_left) / 2
    return shoulder_mid_point, hip_mid_point


def trunk_angle(hip_mid, shoulder_mid, Ds):
    """
    Calculate the trunk angle
    :param hip_mid: Hip mid-point
    :param shoulder_mid: Shoulder mid-point
    :return: Trunk angle
    """
    trunk_angle_angle = 90 - calculate_angle(hip_mid, shoulder_mid, Ds)
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
    shoulder_mid_point, hip_mid_point = mid_vector(shoulder_left, shoulder_right, hip_left, hip_right)
    HH = hip_right - hip_left
    CB = shoulder_mid_point - hip_mid_point
    SS = shoulder_right - shoulder_left
    trunk_twist_angle = calculate_angle(orthogonal_projection(HH, CB), orthogonal_projection(SS, CB))
    return trunk_twist_angle


def trunk_bending(Y, CB, Dh):
    """
    Calculate the trunk bending
    :param Y: Y vector
    :param CB: CB vector
    :param Dh: Dh vector
    :return: Trunk bending
    """
    trunk_bending_angle = calculate_angle(Y, orthogonal_projection(CB, Dh))
    return trunk_bending_angle


def knee_angle(hip_left, knee_left, ankle_left):
    """
    Calculate the knee angle
    :param hip_left: Left hip
    :param knee_left: Left knee
    :param ankle_left: Left ankle
    :return: Knee angle
    """
    knee_angle_angle = calculate_angle(hip_left, knee_left, ankle_left)
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
    shoulder_angle_angle = (calculate_angle(orthogonal_projection(ES, np.cross(Ds, CB)), CB) *
                            np.sign(np.dot(SE, Ds)))
    return shoulder_angle_angle


def elbow_angle(shoulder_left, elbow_left, wrist_left):
    """
    Calculate the elbow angle
    :param shoulder_left: Left shoulder
    :param elbow_left: Left elbow
    :param wrist_left: Left wrist
    :return: Elbow angle
    """
    elbow_angle_angle = calculate_angle(shoulder_left, elbow_left, wrist_left)
    return elbow_angle_angle


def ankle_angle(knee_left, ankle_left, toe_left):
    """
    Calculate the ankle angle
    :param knee_left: Left knee
    :param ankle_left: Left ankle
    :param toe_left: Left toe
    :return: Ankle angle
    """
    ankle_angle = calculate_angle(knee_left, ankle_left, toe_left)
    return ankle_angle

def calculate_mpsae(target, prediction, procrustes=True, Y_target=np.array([0, 1, 0]), Y_prediction=np.array([0, 1, 0])):
    """
    Calculate the mean per segment angle error.
    :param target: Ground truth 3D joint positions, dictionary with keys as joint names
    :param prediction: Predicted 3D joint positions, dictionary with keys as joint names
    :param procrustes: If Procrustes alignment should be done (Default True)
    :param Y_target: Global vertical axis for target keypoints (Default 0, 1, 0])
    :param Y_prediction: Global vertical axis for prediction keypoints (Default 0, 1, 0])
    :return: Mean error, standard deviation of error, segment-wise errors
    """
    assert len(prediction) == len(target), "The shape of prediction and target must match."

    # Align data using Procrustes
    if procrustes:
        target, prediction, error_count = align_procrustes(target, prediction)

    # Midpoint calculation
    target_shoulder_mid, target_hip_mid = mid_vector(target['shoulder_right'], target['hip_right'], target['hip_left'], target['hip_right'])
    prediction_shoulder_mid, prediction_hip_mid = mid_vector(prediction['shoulder_right'], prediction['hip_right'], prediction['hip_left'], prediction['hip_right'])

    # Helper vectors
    target_CB = target_shoulder_mid - target_hip_mid
    prediction_CB = prediction_shoulder_mid - prediction_hip_mid
    target_Ds, target_Dh = help_vector(target_hip_mid, target_shoulder_mid, target['shoulder_left'],
                                       target['shoulder_right'], target['hip_right'], target['hip_left'],
                                       Y_target)
    prediction_Ds, prediction_Dh = help_vector(prediction_hip_mid, prediction_shoulder_mid, prediction['shoulder_left'],
                                               prediction['shoulder_right'], prediction['hip_right'],
                                               prediction['hip_left'], Y_prediction)

    trunk_twist_error = np.abs(
        trunk_twist(target['hip_left'], target['hip_right'], target['shoulder_left'], target['shoulder_right']) -
        trunk_twist(prediction['hip_left'], prediction['hip_right'], prediction['shoulder_left'],
                    prediction['shoulder_right']))

    trunk_bending_error = np.abs(
        trunk_bending(Y_target, target_CB, target_Dh) -
        trunk_bending(Y_prediction, prediction_CB, prediction_Dh))

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
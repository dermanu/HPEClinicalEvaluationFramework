import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict

def vector_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate the angle between two vectors in 3D.
    :param v1: First vector, shape (..., 3)
    :param v2: Second vector, shape (..., 3)
    :return: Angles in degrees, shape (...)
    """
    v1_norm = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    v2_norm = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)
    dot_product = np.sum(v1_norm * v2_norm, axis=-1)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Clip to handle numerical issues
    angles_rad = np.arccos(dot_product)
    return np.degrees(angles_rad)

def orthogonal_projection(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Calculate the orthogonal projection of a vector onto a plane defined by a normal vector.
    :param vector: The vector to be projected, shape (n,)
    :param normal: The normal vector of the plane, shape (n,)
    :return: The orthogonal projection of the vector onto the plane, shape (n,)
    """
    return vector - np.dot(vector, normal) / np.dot(normal, normal) * normal

def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate the midpoint between two points.
    :param a: First point, shape (3,)
    :param b: Second point, shape (3,)
    :return: Midpoint, shape (3,)
    """
    return (a + b) / 2

def help_vector(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Calculate a help vector from four points.
    :param A: Point A, shape (3,)
    :param B: Point B, shape (3,)
    :param C: Point C, shape (3,)
    :param D: Point D, shape (3,)
    :return: Help vector, shape (3,)
    """
    return np.cross(A - B, C - D)

def angle_between_points(joint_a: np.ndarray, joint_b: np.ndarray, joint_c: np.ndarray) -> np.ndarray:
    """
    Calculate the angle between two segments in 3D.
    :param joint_a: [frames, joint, 3]
    :param joint_b: [frames, joint, 3]
    :param joint_c: [frames, joint, 3]
    :return: Joint angles, [frames, joint]
    """
    AB = joint_b - joint_a
    BC = joint_c - joint_b
    return vector_angle(AB, BC)

def trunk_twist(hip_left: np.ndarray, hip_right: np.ndarray, shoulder_left: np.ndarray, shoulder_right: np.ndarray) -> np.ndarray:
    """
    Calculate the trunk twist angle.
    :param hip_left: Left hip, shape (3,)
    :param hip_right: Right hip, shape (3,)
    :param shoulder_left: Left shoulder, shape (3,)
    :param shoulder_right: Right shoulder, shape (3,)
    :return: Trunk twist angle, shape ()
    """
    shoulder_mid = midpoint(shoulder_left, shoulder_right)
    hip_mid = midpoint(hip_left, hip_right)
    HH = hip_right - hip_left
    CB = shoulder_mid - hip_mid
    SS = shoulder_right - shoulder_left
    return vector_angle(orthogonal_projection(HH, CB), orthogonal_projection(SS, CB))

def calculate_mpsae(target: Dict[str, np.ndarray], prediction: Dict[str, np.ndarray], procrustes: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the mean per segment angle error.
    :param target: Ground truth 3D joint positions, dictionary with keys as joint names
    :param prediction: Predicted 3D joint positions, dictionary with keys as joint names
    :param procrustes: Whether to use procrustes alignment
    :return: Mean error, standard deviation of error, segment-wise errors
    """
    target_Y = np.array([0, 1, 0])
    prediction_Y = np.array([0, 1, 0])

    target_hip_mid = midpoint(target['hip_left'], target['hip_right'])
    prediction_hip_mid = midpoint(prediction['hip_left'], prediction['hip_right'])
    target_shoulder_mid = midpoint(target['shoulder_left'], target['shoulder_right'])
    prediction_shoulder_mid = midpoint(prediction['shoulder_left'], prediction['shoulder_right'])
    target_CB = target_shoulder_mid - target_hip_mid
    prediction_CB = prediction_shoulder_mid - prediction_hip_mid
    target_Ds = help_vector(target_hip_mid, target_shoulder_mid, target['shoulder_left'], target['shoulder_right'])
    prediction_Ds = help_vector(prediction_hip_mid, prediction_shoulder_mid, prediction['shoulder_left'], prediction['shoulder_right'])
    target_Dh = help_vector(target_Y, target['hip_left'], target['hip_right'])
    prediction_Dh = help_vector(prediction_Y, prediction['hip_left'], prediction['hip_right'])

    errors = {
        'trunk_twist': trunk_twist(target['hip_left'], target['hip_right'], target['shoulder_left'], target['shoulder_right']) -
                       trunk_twist(prediction['hip_left'], prediction['hip_right'], prediction['shoulder_left'], prediction['shoulder_right']),
        'trunk_bending': vector_angle(target_Y, orthogonal_projection(target_CB, target_Dh)) -
                         vector_angle(prediction_Y, orthogonal_projection(prediction_CB, prediction_Dh)),
        'trunk_angle': vector_angle(target_hip_mid, target_shoulder_mid, target_Ds) -
                       vector_angle(prediction_hip_mid, prediction_shoulder_mid, prediction_Ds),
        'knee_left': angle_between_points(target['hip_left'], target['knee_left'], target['ankle_left']) -
                     angle_between_points(prediction['hip_left'], prediction['knee_left'], prediction['ankle_left']),
        'knee_right': angle_between_points(target['hip_right'], target['knee_right'], target['ankle_right']) -
                      angle_between_points(prediction['hip_right'], prediction['knee_right'], prediction['ankle_right']),
        'shoulder_left': angle_between_points(target['elbow_left'], target['shoulder_left'], target_Ds, target_CB) -
                         angle_between_points(prediction['elbow_left'], prediction['shoulder_left'], prediction_Ds, prediction_CB),
        'shoulder_right': angle_between_points(target['elbow_right'], target['shoulder_right'], target_Ds, target_CB) -
                          angle_between_points(prediction['elbow_right'], prediction['shoulder_right'], prediction_Ds, prediction_CB),
        'elbow_left': angle_between_points(target['shoulder_left'], target['elbow_left'], target['wrist_left']) -
                      angle_between_points(prediction['shoulder_left'], prediction['elbow_left'], prediction['wrist_left']),
        'elbow_right': angle_between_points(target['shoulder_right'], target['elbow_right'], target['wrist_right']) -
                       angle_between_points(prediction['shoulder_right'], prediction['elbow_right'], prediction['wrist_right']),
        'ankle_left': angle_between_points(target['knee_left'], target['ankle_left'], target['toe_left']) -
                      angle_between_points(prediction['knee_left'], prediction['ankle_left'], prediction['toe_left']),
        'ankle_right': angle_between_points(target['knee_right'], target['ankle_right'], target['toe_right']) -
                       angle_between_points(prediction['knee_right'], prediction['ankle_right'], prediction['toe_right']),
    }

    single_segment_error = np.abs(np.array(list(errors.values())))
    single_segment_mean_error = np.mean(single_segment_error, axis=0)
    single_segment_std_error = np.std(single_segment_error, axis=0)

    return single_segment_mean_error, single_segment_std_error, single_segment_error

def calculate_r2(target: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate the coefficient of determination R^2
    :param target: Actual values
    :param predicted: Predicted values
    :return: R^2
    """
    ss_res = np.sum((target - predicted) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    return 1 - (ss_res / ss_tot)

def plot_r2(values: np.ndarray, segment_names: list[str]) -> None:
    """
    Plot R^2 values for each segment.
    :param values: list of R^2 values
    :param segment_names: names of the segments

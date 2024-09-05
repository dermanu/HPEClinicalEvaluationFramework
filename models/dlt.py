import numpy as np
from scipy import linalg


def make_homogeneous_rep_matrix(R, t):
    """
    Creates a 4x4 homogeneous representation matrix from rotation and translation vectors.

    Parameters:
    R (ndarray): 3x3 rotation matrix.
    t (ndarray): 3x1 translation vector.

    Returns:
    ndarray: 4x4 homogeneous representation matrix.
    """
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    return P


def DLT(projection_matrices, points_2d):
    """
    Performs Direct Linear Transform (DLT) to triangulate 3D points from multiple camera views.

    Parameters:
    projection_matrices (list of ndarray): List of 3x4 projection matrices for each camera.
    points_2d (ndarray): Array of 2D points with shape (n_keypoints, n_cams, 2).

    Returns:
    ndarray: Array of triangulated 3D points with shape (n_keypoints, 3).
    """
    n_keypoints = points_2d.shape[0]
    n_views = points_2d.shape[1]
    points_3d = []

    # Loop over each keypoint
    for k in range(n_keypoints):
        A = []

        # For each view/camera, add two equations to A for the current keypoint
        for i in range(n_views):
            P = projection_matrices[i]
            point = points_2d[k, i]

            # Create the two rows for this point/camera combination
            A.append(point[1] * P[2, :] - P[1, :])
            A.append(P[0, :] - point[0] * P[2, :])

        A = np.array(A)

        # Perform Singular Value Decomposition (SVD) to solve the system
        U, s, Vh = linalg.svd(A)

        # Normalize and store the triangulated 3D point
        X = Vh[-1]
        points_3d.append(X[:3] / X[3])

    return np.array(points_3d)


def read_camera_parameters(camera_id, folder='camera_parameters/'):
    """
    Reads the camera intrinsic matrix and distortion coefficients from a file.

    Parameters:
    camera_id (int): The ID of the camera.
    folder (str): Folder path where the camera parameter files are stored.

    Returns:
    tuple: Camera intrinsic matrix and distortion coefficients.
    """
    with open(f'{folder}c{camera_id}.dat', 'r') as inf:
        cmtx = [list(map(float, inf.readline().split())) for _ in range(3)]
        inf.readline()  # Skip a line
        dist = list(map(float, inf.readline().split()))

    return np.array(cmtx), np.array(dist)


def read_rotation_translation(camera_id, folder='camera_parameters/'):
    """
    Reads the camera rotation matrix and translation vector from a file.

    Parameters:
    camera_id (int): The ID of the camera.
    folder (str): Folder path where the rotation and translation files are stored.

    Returns:
    tuple: Rotation matrix and translation vector.
    """
    with open(f'{folder}rot_trans_c{camera_id}.dat', 'r') as inf:
        inf.readline()  # Skip a line
        rot = [list(map(float, inf.readline().split())) for _ in range(3)]
        inf.readline()  # Skip a line
        trans = [list(map(float, inf.readline().split())) for _ in range(3)]

    return np.array(rot), np.array(trans)


def convert_to_homogeneous(pts):
    """
    Converts points to homogeneous coordinates.

    Parameters:
    pts (ndarray): Array of 2D or 3D points.

    Returns:
    ndarray: Homogeneous coordinates.
    """
    pts = np.array(pts)
    if pts.ndim > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis=1)
    return np.concatenate([pts, [1]], axis = 0)


def get_projection_matrix(camera_id, folder='camera_parameters/'):
    """
    Computes the projection matrix for a given camera.

    Parameters:
    camera_id (int): The ID of the camera.
    folder (str): Folder path where camera parameters are stored.

    Returns:
    ndarray: 3x4 projection matrix.
    """
    cmtx, dist = read_camera_parameters(camera_id, folder)
    rvec, tvec = read_rotation_translation(camera_id, folder)

    # Calculate projection matrix
    projection_matrix = cmtx @ make_homogeneous_rep_matrix(rvec, tvec)[:3, :]
    return projection_matrix
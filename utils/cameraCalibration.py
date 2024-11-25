import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml


def read_camera_parameters(camera_id, base_path='./utils/'):
    file_path = base_path + 'sba-profile.txt'
    with open(file_path, 'r') as inf:
        lines = inf.readlines()

    if camera_id >= len(lines):
        raise IndexError(f"Camera ID {camera_id} is out of range. There are only {len(lines)} cameras.")

    line = lines[camera_id].split()
    fx = float(line[0])
    cx = float(line[1])
    cy = float(line[2])
    aspect_ratio = float(line[3])  # Aspect ratio
    skew = float(line[4])  # Skew

    # Calculate fy using the aspect ratio
    fy = fx / aspect_ratio

    cmtx = [[fx, skew, cx],
            [0, fy, cy],
            [0, 0, 1]]

    dist = [float(line[5]), float(line[6]), float(line[7]), float(line[8]), float(line[9])]

    return np.array(cmtx), np.array(dist)


def quaternion_to_rotation_matrix(q):
    r = R.from_quat(q)
    return r.as_matrix()


def read_rotation_translation(camera_id, base_path='./utils/'):
    file_path = base_path + 'sba-profile.txt'
    with open(file_path, 'r') as inf:
        lines = inf.readlines()

    if camera_id >= len(lines):
        raise IndexError(f"Camera ID {camera_id} is out of range. There are only {len(lines)} cameras.")

    line = lines[camera_id].split()
    q = [float(line[10]), float(line[11]), float(line[12]), float(line[13])]
    R = quaternion_to_rotation_matrix(q)
    t = [float(line[14]), float(line[15]), float(line[16])]

    return R, np.array(t)


def make_homogeneous_rep_matrix(R, t):
    if R.shape != (3, 3):
        raise ValueError(f"Rotation matrix R has incorrect shape: {R.shape}")
    if t.shape != (3,):
        raise ValueError(f"Translation vector t has incorrect shape: {t.shape}")

    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1
    return P


def get_projection_matrix(camera_ids, noise=False, noise_percent = 0.01, file_path='utils/P_values.yaml'):
    """
    Retrieves and optionally perturbs projection matrices for specified camera IDs.

    Args:
        camera_ids (int or list of int): ID(s) of the cameras whose projection matrices are to be retrieved.
        noise (bool): Whether to add Gaussian noise to the projection matrices.
        noise_percent (float): Percentage of the mean absolute translation used to scale the noise.
        file_path (str): Path to the YAML file containing projection matrices.

    Returns:
        dict: A dictionary mapping camera keys to their (possibly noisy) projection matrices.
    """

    # Object distance: 3.5 m, pixel_size = 4.6 um, f = 1280 mm
    # 1% correspond to 2.5 px
    if isinstance(camera_ids, int):
        camera_ids = [camera_ids]

    projection_matrices = {}
    with open(file_path, 'r') as yaml_file:
        P_dict = yaml.safe_load(yaml_file)

    for camera_id in camera_ids:
        cam_key = f"Camera_{camera_id}"
        if cam_key not in P_dict:
            raise KeyError(f"Camera ID {camera_id} not found in file {file_path}")

        P = np.array(P_dict[cam_key])
        if noise:
            rg = np.random.default_rng()

            P[:, :3] += rg.normal(loc=0, scale=np.abs(np.mean(P[:, :3])) * noise_percent, size=P[:, :3].shape)
            P[:, 3] += rg.normal(loc=0, scale=np.abs(np.mean(P[:, 3])) * noise_percent, size=P[:, 3].shape)

        projection_matrices[camera_id] = P
    return projection_matrices

def plot_cameras(camera_positions, camera_orientations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for pos, orient in zip(camera_positions, camera_orientations):
        ax.scatter(pos[0], pos[1], pos[2], c='r', marker='o')
        ax.quiver(pos[0], pos[1], pos[2], orient[0, 2], orient[1, 2], orient[2, 2], length=0.1, color='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.view_init(elev=90, azim=0)  # Top-down view
    plt.show()


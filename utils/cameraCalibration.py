import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def get_projection_matrix(camera_ids, noise=False):
    if isinstance(camera_ids, int):
        camera_ids = [camera_ids]

    projection_matrices = {}
    camera_positions = []
    camera_orientations = []

    for camera_id in camera_ids:
        try:
            cmtx, dist = read_camera_parameters(camera_id)
            R, t = read_rotation_translation(camera_id)

            if noise:
                rg = np.random.default_rng(1)
                R = R + np.random.normal(size=R.shape, loc=np.mean(R) * 0.02, scale=np.std(R) * 0.02)
                t = t + np.random.normal(size=t.shape, loc=np.mean(t) * 0.02, scale=np.abs(np.mean(t)) * 0.02)

            P = cmtx @ make_homogeneous_rep_matrix(R, t)[:3, :]
            projection_matrices[camera_id] = P
            camera_positions.append(t)
            camera_orientations.append(R)
        except (IndexError, ValueError) as e:
            print(f"Error processing camera {camera_id}: {e}")

    return projection_matrices, camera_positions, camera_orientations


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

# Example usage:
#camera_ids = [0, 1, 2, 3, 4, 5]
#projection_matrices, camera_positions, camera_orientations = get_projection_matrix(camera_ids, noise=False)
#plot_cameras(camera_positions, camera_orientations)

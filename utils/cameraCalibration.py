"""
Helper functions to read the camera calibration matrices (P_values.yaml) and augment them if needed.
"""

import numpy as np
import yaml


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
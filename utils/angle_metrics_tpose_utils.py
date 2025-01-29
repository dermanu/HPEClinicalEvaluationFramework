"""
Calculation of T-Pose for joint angle calculation based on: https://github.com/TemugeB/joint_angles_calculate
"""

import numpy as np

def get_R_x(theta):
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
    return R

def get_R_y(theta):
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0,  np.cos(theta)]])
    return R

def get_R_z(theta):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return R


#calculate rotation matrix to take A vector to B vector
def Get_R(A, B, epsilon=1e-6):

    #get unit vectors
    uA = A / np.linalg.norm(A)
    uB = B / np.linalg.norm(B)

    #get products
    dotprod = np.dot(uA, uB)
    crossprod = np.cross(uA, uB)
    crossprod_mag = np.linalg.norm(crossprod)

    if crossprod_mag < epsilon:
        if dotprod > 0:
            # A and B are in the same direction
            return np.eye(3)
        else:
            # A and B are in opposite directions, 180-degree rotation
            orthogonal_vector = np.array([1, 0, 0]) if np.abs(uA[0]) < 1 - epsilon else np.array([0, 1, 0])
            crossprod = np.cross(uA, orthogonal_vector)
            crossprod = crossprod / np.linalg.norm(crossprod)
            R = np.eye(3) - 2 * np.outer(crossprod, crossprod)
            return R

    #get new unit vectors
    u = uA
    v = uB - dotprod * uA
    v = v / np.linalg.norm(v)
    w = crossprod / crossprod_mag

    #get change of basis matrix
    C = np.array([u, v, w])

    #get rotation matrix in new basis
    R_uvw = np.array([[dotprod, -crossprod, 0],
                      [crossprod, dotprod, 0],
                      [0, 0, 1]])

    #full rotation matrix
    R = C.T @ R_uvw @ C
    return R


#Same calculation as above using a different formalism
def Get_R2(A, B, epsilon=1e-6):
    uA = A / np.linalg.norm(A)
    uB = B / np.linalg.norm(B)

    v = np.cross(uA, uB)
    s = np.linalg.norm(v)
    c = np.dot(uA, uB)

    if s < epsilon:
        if c > 0:
            # A and B are in the same direction
            return np.eye(3)
        else:
            # A and B are in opposite directions, 180-degree rotation
            orthogonal_vector = np.array([1, 0, 0]) if np.abs(uA[0]) < 1 - epsilon else np.array([0, 1, 0])
            v = np.cross(uA, orthogonal_vector)
            v = v / np.linalg.norm(v)
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            return np.eye(3) - 2 * vx @ vx

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

    return R

#decomposes given R matrix into rotation along each axis. In this case Rz @ Ry @ Rx
def Decompose_R_ZYX(R, epsilon=1e-6):
    #decomposes as RzRyRx. Note the order: ZYX <- rotation by x first

    if np.abs(R[2, 0]) < 1 - epsilon:
        thetaz = np.arctan2(R[1, 0], R[0, 0])
        thetay = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        thetax = np.arctan2(R[2, 1], R[2, 2])
    else:
        # Gimbal lock case: thetay is +/- 90 degrees
        thetaz = 0
        if R[2, 0] < 0:
            thetay = np.pi / 2
            thetax = np.arctan2(R[0, 1], R[1, 1])
        else:
            thetay = -np.pi / 2
            thetax = np.arctan2(-R[0, 1], -R[1, 1])

    return thetaz, thetay, thetax

def Decompose_R_ZXY(R, epsilon=1e-6):
    if np.abs(R[2, 1]) < 1 - epsilon:
        thetaz = np.arctan2(-R[0, 1], R[1, 1])
        thetay = np.arctan2(-R[2, 0], R[2, 2])
        thetax = np.arctan2(R[2, 1], np.sqrt(R[2, 0] ** 2 + R[2, 2] ** 2))
    else:
        # Gimbal lock case: thetax is +/- 90 degrees
        thetay = 0
        if R[2, 1] < 0:
            thetax = np.pi / 2
            thetaz = np.arctan2(R[1, 0], R[0, 0])
        else:
            thetax = -np.pi / 2
            thetaz = np.arctan2(-R[1, 0], -R[0, 0])

    return thetaz, thetay, thetax
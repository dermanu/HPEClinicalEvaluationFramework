import numpy as np


def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1
    return P


def read_camera_parameters(camera_id):
    # get the projection matrices

    inf = open('camera_parameters/S1/c' + str(camera_id) + '.dat', 'r')

    cmtx = []
    dist = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    return np.array(cmtx), np.array(dist)


def read_rotation_translation(camera_id, savefolder='camera_parameters/S1/'):
    inf = open(savefolder + 'rot_trans_c' + str(camera_id) + '.dat', 'r')

    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)


def get_projection_matrix(camera_id, noise=False):
    # read camera parameters
    cmtx, dist = read_camera_parameters(camera_id)
    rvec, tvec = read_rotation_translation(camera_id)
    print(rvec)
    print(tvec)

    if noise:
        # add noise to rotation and translation
        # Add normal distributed noise with mean of 2 px and std of 1px (depends on results of calculation above)
        rvec = rvec + np.random.normal(size=rvec.shape, loc=np.mean(rvec) * 0.02, scale=np.std(rvec) * 0.02)
        tvec = tvec + np.random.normal(size=tvec.shape, loc=np.mean(tvec) * 0.02, scale=np.mean(tvec) * 0.02)

    print(np.std(rvec) * 0.1)
    print(rvec)
    print(tvec)

    # calculate projection matrix
    P = cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3, :]
    return P

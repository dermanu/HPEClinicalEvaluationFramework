# Spatial and spatiotemporal filter needs at least a window size of 20 samples (0.67 sec)
# Class camera optim_points. Dependend on spatial=True constraints are applied, spatial = False, only 3D filtering is done

import numpy as np
from scipy.sparse import dok_matrix
from scipy import optimize
import scipy
from numba import jit
import time
import tensorflow as tf
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


# Some helper function... unsure what it does
def resample_points(imgp, extra=None, n_samp=25):
    # if extra is not None:
    #     return resample_points_extra(imgp, extra, n_samp)

    n_cams = imgp.shape[0]
    good = ~np.isnan(imgp[:, :, 0])
    ixs = np.arange(imgp.shape[1])

    num_cams = np.sum(~np.isnan(imgp[:, :, 0]), axis=0)

    include = set()

    for i in range(n_cams):
        for j in range(i + 1, n_cams):
            subset = good[i] & good[j]
            n_good = np.sum(subset)
            if n_good > 0:
                ## pick points, prioritizing points seen by more cameras
                arr = np.copy(num_cams[subset]).astype('float64')
                arr += np.random.random(size=arr.shape)
                picked_ix = np.argsort(-arr)[:n_samp]
                picked = ixs[subset][picked_ix]
                include.update(picked)

    final_ixs = sorted(include)
    newp = imgp[:, final_ixs]
    extra = subset_extra(extra, final_ixs)
    return newp, extra


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def subset_extra(extra, ixs):
    if extra is None:
        return None

    new_extra = {
        'objp': extra['objp'][ixs],
        'ids': extra['ids'][ixs],
        'rvecs': extra['rvecs'][:, ixs],
        'tvecs': extra['tvecs'][:, ixs]
    }
    return new_extra


# Simple linear interpolation (as subsequent smoothing no polynomial interpolation needed)
def interpolate_data(vals):
    nans, ix = nan_helper(vals)
    out = np.copy(vals)
    try:
        out[nans] = np.interp(ix(nans), ix(~nans), vals[~nans])
    except ValueError:
        out[:] = 0
    return out


def upsample_data(values, times, der=0):
    xnew = np.arange(0, len(values) - 1, 1 / times)
    xi = np.arange(0, len(values))
    yi = np.array(values)
    try:
        P = scipy.interpolate.Akima1DInterpolator(xi, yi, axis=0)
    except TypeError:
        # Scipy earlier than 0.17.0 missing axis
        P = scipy.interpolate.Akima1DInterpolator(xi, yi)
    if der == 0:
        return P(xnew)
    elif scipy.interpolate._isscalar(der):
        return P(xnew, der=der)
    else:
        return [P(xnew, nu) for nu in der]


# Median filter with a specific window size (size)
def medfilt_data(values, size=15):
    padsize = size + 5
    vpad = np.pad(values, (padsize, padsize), mode='reflect')
    # vpadf = signal.medfilt(vpad, kernel_size=size)
    vpadf = scipy.ndimage.median_filter(vpad, size=size)  # More efficient than signal.medfilt
    return vpadf[padsize:-padsize]


# Kalman smoothing (https://pubmed.ncbi.nlm.nih.gov/35746410/
def rts_smoother(Xs, Ps, F, Q):
    n, dim_x, _ = Xs.shape

    # smoother gain
    K = np.zeros((n, dim_x, dim_x))
    x, P, Pp = Xs.copy(), Ps.copy(), Ps.copy

    for k in range(n - 2, -1, -1):
        Pp[k] = F @ P[k] @ F.T + Q  # predicted covariance

        K[k] = P[k] @ F.T @ inv(Pp[k])
        x[k] += K[k] @ (x[k + 1] - (F @ x[k]))
        P[k] += K[k] @ (P[k + 1] - Pp[k]) @ K[k].T
    return (x, P, K, Pp)


def kalfilt_data(values, noise, Q=0.01):
    # Maybe initializing Kalman filter earlier (doesn't need to be done every iteration)
    fk = KalmanFilter(dim_x=2, dim_z=1)
    fk.x = np.array([0., 1.])  # state (x and dx)

    fk.F = np.array([[1., 1.],
                     [0., 1.]])  # state transition matrix

    fk.H = np.array([[1., 0.]])  # Measurement function
    fk.P *= 10.  # covariance matrix
    fk.R = noise  # state uncertainty
    fk.Q = Q_discrete_white_noise(dim=2, dt=1., var=Q)  # process uncertainty
    mu, cov, _, _ = fk.batch_filter(values)
    M, P, C, _ = fk.rts_smoother(mu, cov)
    return M


# Spline smoothing (https://stackoverflow.com/questions/45179024/scipy-bspline-fitting-in-python and
# https://pubmed.ncbi.nlm.nih.gov/35746410/)
def bsplinefilt_data(values, k=3):
    t, c, k = scipy.interpolate.splrep(np.arange(len(values)), values, k=k, s=0)
    spline = scipy.interpolate.BSpline(t=t, c=c, k=k, extrapolate=False)
    return spline(np.arange(len(values)))


class CameraGroup:
    def __init__(self, cameras):
        self.cameras = cameras

    def optim_points(self, p3ds,
                     constraints=[],
                     constraints_weak=[],
                     scale_smooth=3,
                     scale_length=2, scale_length_weak=0.5,
                     n_deriv_smooth=1, spatial=False, verbose=True, filter='median', upsampling=False):
        """
        Take in an array of 2D points of shape CxNxJx2,
        an array of 3D points of shape NxJx3,
        and an array of constraints of shape Kx2, where
        C: number of camera
        N: number of frames
        J: number of joints
        K: number of constraints
        This function creates an optimized array of 3D points of shape NxJx3.
        Example constraints:
        constraints = [[0, 1], [1, 2], [2, 3]]
        (meaning that lengths of segments 0->1, 1->2, 2->3 are all constant)
        """
        dims = [self.cameras, p3ds.shape[0], p3ds.shape[1]]
        n_cams = dims[0]
        n_frames = dims[1]
        n_joints = dims[2]
        constraints = np.array(constraints)
        constraints_weak = np.array(constraints_weak)

        p3ds_intp = np.apply_along_axis(interpolate_data, 0, p3ds)

        if filter == 'spline':
            p3ds_med = np.apply_along_axis(bsplinefilt_data, 0, p3ds_intp, k=3)
        elif filter == 'kalman':
            p3ds_med = np.apply_along_axis(kalfilt_data, 0, p3ds_intp, size=7)
        else:
            p3ds_med = np.apply_along_axis(medfilt_data, 0, p3ds_intp, size=7)

        default_smooth = 1.0 / np.mean(np.abs(np.diff(p3ds_med, axis=0)))
        scale_smooth_full = scale_smooth * default_smooth

        t1 = time.time()

        x0 = self._initialize_params_triangulation(p3ds_intp, constraints, constraints_weak)

        x0[~np.isfinite(x0)] = 0
        jac = self._jac_sparsity_triangulation(dims, constraints, constraints_weak, n_deriv_smooth)

        if spatial:
            opt2 = optimize.least_squares(self._error_fun_triangulation,
                                          x0=x0, jac_sparsity=jac,
                                          loss='linear',
                                          ftol=1e-3,
                                          max_nfev=5,
                                          diff_step=1.5,
                                          tr_solver='lsmr',
                                          verbose=2 * verbose,
                                          args=(dims,
                                                constraints,
                                                constraints_weak,
                                                scale_smooth_full,
                                                scale_length,
                                                scale_length_weak,
                                                n_deriv_smooth))

            p3ds_new2 = opt2.x[:p3ds.size].reshape(p3ds.shape)
        else:
            p3ds_new2 = p3ds_med

        t2 = time.time()

        if upsampling:
            print('Start upsampling')
            t1up = time.time()
            p3ds_new2 = np.apply_along_axis(upsample_data, 0, p3ds_new2, times=4)
            t2up = time.time()
            if verbose:
                print('Upsampling took {:.2f} ms per frame'.format((t2up - t1up)*1000))
            upsamling_time = t2up - t1up / p3ds.shape[0]
        else:
            upsamling_time = 0

        if verbose:
            print('Optimization took {:.2f} ms per frame.'.format(((t2 - t1) / p3ds.shape[0])*1000))
        total_opti_time = (t2 - t1) / p3ds.shape[0]


        return p3ds_new2, [upsamling_time, total_opti_time]

    def _initialize_params_triangulation(self, p3ds,
                                         constraints=[],
                                         constraints_weak=[]):
        joint_lengths = np.empty(len(constraints), dtype='float64')
        joint_lengths_weak = np.empty(len(constraints_weak), dtype='float64')

        for cix, (a, b) in enumerate(constraints):
            lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
            joint_lengths[cix] = np.median(lengths)

        for cix, (a, b) in enumerate(constraints_weak):
            lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
            joint_lengths_weak[cix] = np.median(lengths)

        all_lengths = np.hstack([joint_lengths, joint_lengths_weak])
        med = np.median(all_lengths)
        if med == 0:
            med = 1e-3

        mad = np.median(np.abs(all_lengths - med))

        joint_lengths[joint_lengths == 0] = med
        joint_lengths_weak[joint_lengths_weak == 0] = med
        joint_lengths[joint_lengths > med + mad * 5] = med
        joint_lengths_weak[joint_lengths_weak > med + mad * 5] = med

        return np.hstack([p3ds.ravel(), joint_lengths, joint_lengths_weak])

    def _jac_sparsity_triangulation(self, dims,
                                    constraints=[],
                                    constraints_weak=[],
                                    n_deriv_smooth=1):
        n_cams = dims[0]
        n_frames = dims[1]
        n_joints = dims[2]
        n_constraints = len(constraints)
        n_constraints_weak = len(constraints_weak)

        # p2ds_flat = p2ds.reshape((n_cams, -1, 2))

        # point_indices = np.zeros(p2ds_flat.shape, dtype='int32')
        # for i in range(p2ds_flat.shape[1]):
        #    point_indices[:, i] = i

        point_indices_3d = np.arange(n_frames * n_joints) \
            .reshape((n_frames, n_joints))

        # good = ~np.isnan(p2ds_flat)
        # n_errors_reproj = np.sum(good)
        n_errors_smooth = (n_frames - n_deriv_smooth) * n_joints * 3
        n_errors_lengths = n_constraints * n_frames
        n_errors_lengths_weak = n_constraints_weak * n_frames

        n_errors = n_errors_smooth + n_errors_lengths + n_errors_lengths_weak

        n_3d = n_frames * n_joints * 3
        n_params = n_3d + n_constraints + n_constraints_weak

        # point_indices_good = point_indices[good]

        A_sparse = dok_matrix((n_errors, n_params), dtype='int16')

        # constraints for re-projection errors
        # ix_reproj = np.arange(n_errors_reproj)
        # for k in range(3):
        #    A_sparse[ix_reproj, point_indices_good * 3 + k] = 1

        # sparse constraints for smoothness in time
        frames = np.arange(n_frames - n_deriv_smooth)
        for j in range(n_joints):
            for n in range(n_deriv_smooth + 1):
                pa = point_indices_3d[frames, j]
                pb = point_indices_3d[frames + n, j]
                for k in range(3):
                    A_sparse[pa * 3 + k, pb * 3 + k] = 1

        ## -- strong constraints --
        # joint lengths should change with joint lengths errors
        start = n_errors_smooth
        frames = np.arange(n_frames)
        for cix, (a, b) in enumerate(constraints):
            A_sparse[start + cix * n_frames + frames, n_3d + cix] = 1

        # points should change accordingly to match joint lengths too
        frames = np.arange(n_frames)
        for cix, (a, b) in enumerate(constraints):
            pa = point_indices_3d[frames, a]
            pb = point_indices_3d[frames, b]
            for k in range(3):
                A_sparse[start + cix * n_frames + frames, pa * 3 + k] = 1
                A_sparse[start + cix * n_frames + frames, pb * 3 + k] = 1

        ## -- weak constraints --
        # joint lengths should change with joint lengths errors
        start = n_errors_smooth + n_errors_lengths
        frames = np.arange(n_frames)
        for cix, (a, b) in enumerate(constraints_weak):
            A_sparse[start + cix * n_frames + frames, n_3d + n_constraints + cix] = 1

        # points should change accordingly to match joint lengths too
        frames = np.arange(n_frames)
        for cix, (a, b) in enumerate(constraints_weak):
            pa = point_indices_3d[frames, a]
            pb = point_indices_3d[frames, b]
            for k in range(3):
                A_sparse[start + cix * n_frames + frames, pa * 3 + k] = 1
                A_sparse[start + cix * n_frames + frames, pb * 3 + k] = 1

        return A_sparse

    # @jit(forceobj=True, parallel=True, nogil=True)
    @jit(forceobj=True, parallel=True)
    def _error_fun_triangulation(self, params, dims,
                                 constraints=[],
                                 constraints_weak=[],
                                 scale_smooth=10000,
                                 scale_length=1,
                                 scale_length_weak=0.2,
                                 n_deriv_smooth=1):

        n_cams = dims[0]
        n_frames = dims[1]
        n_joints = dims[2]
        n_3d = n_frames * n_joints * 3
        n_constraints = len(constraints)
        n_constraints_weak = len(constraints_weak)

        # load params
        p3ds = params[:n_3d].reshape((n_frames, n_joints, 3))
        joint_lengths = np.array(params[n_3d:n_3d + n_constraints])
        joint_lengths_weak = np.array(params[n_3d + n_constraints:])

        # temporal constraint
        errors_smooth = np.diff(p3ds, n=n_deriv_smooth, axis=0).ravel() * scale_smooth

        # joint length constraint
        errors_lengths = np.empty((n_constraints, n_frames), dtype='float64')
        for cix, (a, b) in enumerate(constraints):
            # Calculate the length between the joints
            lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
            expected = joint_lengths[cix]
            errors_lengths[cix] = 100 * (lengths - expected) / expected
        errors_lengths = errors_lengths.ravel() * scale_length

        errors_lengths_weak = np.empty((n_constraints_weak, n_frames), dtype='float64')
        for cix, (a, b) in enumerate(constraints_weak):
            # Calculate the length between the joints
            lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
            expected = joint_lengths_weak[cix]
            errors_lengths_weak[cix] = 100 * (lengths - expected) / expected
        errors_lengths_weak = errors_lengths_weak.ravel() * scale_length_weak

        return np.hstack([errors_smooth, errors_lengths, errors_lengths_weak])

        #  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,


pose_keypoints_prediction = np.array([16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28, 29, 30, 31, 32, 18, 20, 19, 17])
pose_keypoints_target = np.array([27, 26, 25, 17, 18, 19, 1, 6, 7, 2, 8, 3, 9, 4, 10, 5, 14])
convert = np.array([5, 3, 1, 0, 2, 4, 11, 10, 12, 13, 14, 15])


# pose_keypoints_target = {1: 'RHip', 2: 'RKnee', 3: 'RAnkle', 4: 'RArch', 5: 'RToe',
#                         6: 'LHip', 7: 'LKnee', 8: 'LAnkle', 9: 'LArch', 10: 'LToe',
#                         0: 'Pelvis', 24: 'Neck', 14: 'CHead',
#                         17: 'LShoulder', 18: 'LElbow', 19: 'LWrist',
#                         25: 'RShoulder', 26: 'RElbow', 27: 'RWrist'}

# pose_keypoints_target = {0: 'RHip', 2: 'RKnee', 3: 'RAnkle', 4: 'RArch', 5: 'RToe',
#                         6: 'LHip', 7: 'LKnee', 8: 'LAnkle', 9: 'LArch', 10: 'LToe',
#                         0: 'Pelvis', 24: 'Neck', 14: 'CHead',
#                         17: 'LShoulder', 18: 'LElbow', 19: 'LWrist',
#                         25: 'RShoulder', 26: 'RElbow', 27: 'RWrist'}


def skeleton_fit(prediction, tree, height=1.7, threshold=0.1):
    tree = {
        {7, 0, 0}, {8, 7, 239.70}, {9, 8, 254.46}, {10, 9, 178.07},
        {13, 9, 150.85}, {12, 13, 279.66}, {11, 12, 246.43}, {14, 9, 150.85}, {15, 14, 279.66}, {16, 15, 246.43},
        {3, 7, 139.59}, {2, 3, 448.04}, {1, 2, 436.62}, {4, 7, 139.59}, {5, 4, 448.04}, {6, 5, 436.62}
    }

    tree = {
        {0, 1, 0.146}, {1, 2, 0.186}, {5, 4, 0.146}, {4, 3, 0.186},
        {11, 9, 0.246}, {9, 6, 0.245}, {10, 8, 0.246}, {8, 7, 0.245},
        {2, 6, 0.288}, {3, 7, 0.288},
    }

    output = np.zeros((np.shape(prediction, 1), np.shape(prediction, 2), np.shape(prediction, 3)))

    for j, bone in enumerate(tree):
        joint = bone[1]
        parent = bone[2]
        bone_length_ratio = bone[3]
        bone_length = height * bone_length_ratio
        bone_length_prediction = np.linalg.norm(prediction[:, joint, :] - prediction[:, parent, :], axis=1)
        bone_threshold = bone_length * threshold
        if parent == 0 or bone_length_prediction < bone_length + bone_threshold or bone_length_prediction > bone_length - bone_threshold:
            # If root joint or within threshold
            output[joint] = prediction[joint]
        else:
            vecnorm = (prediction[joint] - prediction[parent]) / np.squeeze(
                np.linalg.norm(prediction[joint] - prediction[parent], 2, 1))
            output[joint] = output[parent] + bone_length * vecnorm
    return output

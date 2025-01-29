#######################################################
## Gap filling and smoothing as post-processing step ##
##            NOT EXTENSIVELY TESTED YET             ##
#######################################################

import numpy as np
import scipy

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def linear_interpolation(input):
    # Simple linear interpolation (as subsequent smoothing no polynomial interpolation needed)
    nans, ix = nan_helper(input)
    out = np.copy(input)
    try:
        out[nans] = np.interp(ix(nans), ix(~nans), input[~nans])
    except ValueError:
        out[:] = 0
    return out


def akima_interpolator(input):
    for i in range(input.shape[1]):
        for j in range(input.shape[2]):
            # Select the slice along the N dimension
            slice_n = input[:, i, j, 0]  # Assuming the first column contains the values to be interpolated

            # Find indices of missing values
            missing_indices = np.isnan(slice_n)

            # Create an Akima1DInterpolator object
            akima_interpolator = scipy.interpolate.Akima1DInterpolator(np.arange(len(slice_n))[~missing_indices], slice_n[~missing_indices])

            # Interpolate missing values
            interpolated_values = akima_interpolator(np.arange(len(slice_n)))

            # Fill in missing values in the original array
            slice_n[missing_indices] = interpolated_values[missing_indices]


def median_smoothing(inputs, size=15):
    # Median filter with a specific window size (size)
    padsize = size + 5
    vpad = np.pad(inputs, (padsize, padsize), mode='reflect')
    # vpadf = signal.medfilt(vpad, kernel_size=size)
    vpadf = scipy.ndimage.median_filter(vpad, size=size)  # More efficient than signal.medfilt

    return vpadf[padsize:-padsize]


def spline_smoothing(inputs, k=3):
    # Spline smoothing (https://stackoverflow.com/questions/45179024/scipy-bspline-fitting-in-python and
    # https://pubmed.ncbi.nlm.nih.gov/35746410/)
    t, c, k = scipy.interpolate.splrep(np.arange(len(inputs)), inputs, k=k, s=0)
    spline = scipy.interpolate.BSpline(t=t, c=c, k=k, extrapolate=False)
    return spline(np.arange(len(inputs)))


def butterworth_smoothing(inputs):
    pass


def postprocess_points(p3ds, filter_type='median', interpolation_type='Akima'):
    """
    Take in an array of 2D points of shape CxNxJx2,
    an array of 3D points of shape NxJx3,
    This function creates an optimized array of 3D points of shape NxJx3.
    """

    if interpolation_type == 'akima':
        p3ds = np.apply_along_axis(akima_interpolator, 0, p3ds)
    else:
        p3ds = np.apply_along_axis(linear_interpolation, 0, p3ds)

    if filter_type == 'butterworth':
        p3ds = np.apply_along_axis(butterworth_smoothing, 0, p3ds, k=3)
    elif filter_type == 'spline':
        p3ds = np.apply_along_axis(spline_smoothing, 0, p3ds, size=7)
    else:
        p3ds = np.apply_along_axis(median_smoothing, 0, p3ds, size=7)

    return p3ds
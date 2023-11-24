# Add noise to extrinsic camera parameters before calculating projection matrix P. Input 3 x 4 x cameras
# Calibration error should be around 2 px, this corresponds to:
# (ErrImage x Distance Camera-Object) / (focal length / pixel size)
def calibration_noise(inputs):
    # Input extrinsic parameter
    R = inputs[0]  # Rotation matrix 3x3
    t = inputs[1]  # Translation vector 3x1

    # Add normal distributed noise with mean of 2 px and std of 1px (depends on results of calculation above)
    noisy_R = R + tf.random.normal(shape=[3, 3], mean=2, stddev=1)
    noisy_t = t + tf.random.normal(shape=[3, 1], mean=2, stddev=1)

    return noisy_R, noisy_t
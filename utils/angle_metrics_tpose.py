import numpy as np
from utils import angle_metrics_tpose_utils as utils
from scipy.signal import medfilt
import gc
import multiprocessing
from scipy.signal import savgol_filter


def convert_to_dictionary(keypoints):
    # Convert keypoints to float32 to save memory
    keypoints = keypoints.astype(np.float32)

    #its easier to manipulate keypoints by joint name
    joint_names = ['right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow',
                   'right_wrist', 'left_wrist', 'right_hip', 'left_hip',
                   'right_knee', 'left_knee', 'right_ankle', 'left_ankle',
                   'right_heel', 'left_heel', 'right_foot_index', 'left_foot_index']

    # Transpose the input to have shape (frames, keypoints, 3)
    keypoints = np.transpose(keypoints, (2, 0, 1))

    # Create a dictionary mapping joint names to their (frames, 3) data
    kpts_dict = {joint: keypoints[:, i, :] for i, joint in enumerate(joint_names)}
    kpts_dict['joints'] = joint_names

    return kpts_dict


def calculate_midpoint(kpt1, kpt2):
    return (kpt1 + kpt2) / 2

def add_hips_and_neck(kpts):
    #we add two new keypoints which are the mid point between the hips and mid point between the shoulders
    kpts['hips'] = calculate_midpoint(kpts['left_hip'], kpts['right_hip'])
    kpts['neck'] = calculate_midpoint(kpts['left_shoulder'], kpts['right_shoulder'])

    kpts['joints'].extend(['hips', 'neck'])

    #define the hierarchy of the joints
    hierarchy = {
        'hips': [],
        'left_hip': ['hips'], 'left_knee': ['left_hip', 'hips'], 'left_ankle': ['left_knee', 'left_hip', 'hips'], 'left_heel': ['left_ankle', 'left_knee', 'left_hip', 'hips'], 'left_foot_index': ['left_heel', 'left_ankle', 'left_knee', 'left_hip', 'hips'],
        'right_hip': ['hips'], 'right_knee': ['right_hip', 'hips'], 'right_ankle': ['right_knee', 'right_hip', 'hips'], 'right_heel': ['right_ankle', 'right_knee', 'right_hip', 'hips'], 'right_foot_index': ['right_heel', 'right_ankle', 'right_knee', 'right_hip', 'hips'],
        'neck': ['hips'],
        'left_shoulder': ['neck', 'hips'], 'left_elbow': ['left_shoulder', 'neck', 'hips'], 'left_wrist': ['left_elbow', 'left_shoulder', 'neck', 'hips'],
        'right_shoulder': ['neck', 'hips'], 'right_elbow': ['right_shoulder', 'neck', 'hips'], 'right_wrist': ['right_elbow', 'right_shoulder', 'neck', 'hips']
    }
    kpts['hierarchy'] = hierarchy
    kpts['root_joint'] = 'hips'

    return kpts


#remove jittery keypoints by applying a median filter along each axis
def median_filter(kpts, window_size=3):
    filtered = kpts.copy()  # A shallow copy should suffice here

    for joint in kpts['joints']:
        joint_kpts = kpts[joint]
        xs = medfilt(joint_kpts[:, 0], window_size)
        ys = medfilt(joint_kpts[:, 1], window_size)
        zs = medfilt(joint_kpts[:, 2], window_size)
        kpts[joint] = np.stack([xs, ys, zs], axis=-1)

    return filtered


def get_bone_lengths(kpts):
    """
    We have to define an initial skeleton pose(T pose).
    In this case we need to known the length of each bone.
    Here we calculate the length of each bone from data
    """

    bone_lengths = {}
    for joint in kpts['joints']:
        if joint == 'hips':
            continue
        parent = kpts['hierarchy'][joint][0]
        _bone = kpts[joint] - kpts[parent]
        _bone_lengths = np.sqrt(np.sum(np.square(_bone), axis=-1))

        _bone_length = np.median(_bone_lengths)
        bone_lengths[joint] = _bone_length

    kpts['bone_lengths'] = bone_lengths
    return


#Here we define the T pose and we normalize the T pose by the length of the hips to neck distance.
def get_base_skeleton(kpts, normalization_bone='neck'):
    body_lengths = kpts['bone_lengths']
    normalization = kpts['bone_lengths'][normalization_bone]

    directions = {
        'left_hip': np.array([1, 0.2, 0]),
        'left_knee': np.array([0, -1, 0]), 'left_ankle': np.array([0, -1, 0]),
        'right_hip': np.array([-1, -0.2, 0]),
        'right_knee': np.array([0, -1, 0]), 'right_ankle': np.array([0, -1, 0]),
        'neck': np.array([0, 1, 0]),
        'left_shoulder': np.array([1, 0, 0]), 'left_elbow': np.array([1, 0, 0]), 'left_wrist': np.array([1, 0, 0]),
        'right_shoulder': np.array([-1, 0, 0]), 'right_elbow': np.array([-1, 0, 0]),
        'right_wrist': np.array([-1, 0, 0]),
        'left_heel': np.array([0, -1, 0]), 'right_heel': np.array([0, -1, 0]),
        'left_foot_index': np.array([0.5, -0.5, 0]), 'right_foot_index': np.array([-0.5, -0.5, 0])
    }

    # Use individual bone lengths for left and right sides
    base_skeleton = {'hips': np.array([0, 0, 0])}

    for side in ['left', 'right']:
        for joint_type in ['hip', 'knee', 'ankle', 'shoulder', 'elbow', 'wrist']:
            joint_name = f'{side}_{joint_type}'
            base_skeleton[joint_name] = directions[joint_name] * (body_lengths[joint_name] / normalization)

    base_skeleton['neck'] = directions['neck'] * (body_lengths['neck'] / normalization)

    kpts['offset_directions'] = directions
    kpts['base_skeleton'] = base_skeleton
    kpts['normalization'] = normalization

    return kpts


#calculate the rotation of the root joint with respect to the world coordinates
def get_hips_position_and_rotation(frame_pos, root_joint = 'hips', root_define_joints = ['left_hip', 'neck']):

    #root position is saved directly
    root_position = frame_pos[root_joint]

    #calculate unit vectors of root joint
    root_u = frame_pos[root_define_joints[0]] - frame_pos[root_joint]
    root_u = root_u/np.sqrt(np.sum(np.square(root_u)))
    root_v = frame_pos[root_define_joints[1]] - frame_pos[root_joint]
    root_v = root_v/np.sqrt(np.sum(np.square(root_v)))
    root_w = np.cross(root_u, root_v)

    #Make the rotation matrix
    C = np.array([root_u, root_v, root_w]).T
    thetaz,thetay, thetax = utils.Decompose_R_ZXY(C)
    root_rotation = np.array([thetaz, thetax, thetay])

    return root_position, root_rotation

#calculate the rotation matrix and joint angles input joint
def get_joint_rotations(joint_name, joints_hierarchy, joints_offsets, frame_rotations, frame_pos):

    _invR = np.eye(3)
    for i, parent_name in enumerate(joints_hierarchy[joint_name]):
        if i == 0: continue
        _r_angles = frame_rotations[parent_name]
        R = utils.get_R_z(_r_angles[0]) @ utils.get_R_x(_r_angles[1]) @ utils.get_R_y(_r_angles[2])
        _invR = _invR@R.T

    b = _invR @ (frame_pos[joint_name] - frame_pos[joints_hierarchy[joint_name][0]])

    _R = utils.Get_R2(joints_offsets[joint_name], b)
    tz, ty, tx = utils.Decompose_R_ZXY(_R)
    joint_rs = np.array([tz, tx, ty])
    #print(np.degrees(joint_rs))

    return joint_rs

#helper function that composes a chain of rotation matrices
def get_rotation_chain(joint, hierarchy, frame_rotations):

    hierarchy = hierarchy[::-1]

    #this code assumes ZXY rotation order
    R = np.eye(3)
    for parent in hierarchy:
        angles = frame_rotations[parent]
        _R = utils.get_R_z(angles[0])@utils.get_R_x(angles[1])@utils.get_R_y(angles[2])
        R = R @ _R

    return R


#calculate the joint angles frame by frame.
def calculate_joint_angles(keypoints):
    # Joints to exclude from the final angles output
    exclude_joints_from_output = ['hips', 'neck', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']

    # Initialize container for joint angles for joints not in exclude list
    for joint in keypoints['joints']:
        if joint not in exclude_joints_from_output:
            keypoints[joint + '_angles'] = []

    num_frames = keypoints['hips'].shape[0]

    for framenum in range(num_frames):
        frame_pos = {joint: keypoints[joint][framenum] for joint in keypoints['joints']}
        root_position, root_rotation = get_hips_position_and_rotation(frame_pos)

        frame_rotations = {'hips': root_rotation}

        # Center the body pose
        for joint in keypoints['joints']:
            frame_pos[joint] -= root_position

        # Ensure the hierarchy is respected in processing
        max_depth = max(len(keypoints['hierarchy'][joint]) for joint in keypoints['joints'])
        for depth in range(1, max_depth + 1):
            for joint in [j for j in keypoints['joints'] if len(keypoints['hierarchy'][j]) == depth]:
                if joint not in frame_rotations:
                    joint_rs = get_joint_rotations(joint, keypoints['hierarchy'], keypoints['offset_directions'],
                                                   frame_rotations, frame_pos)
                    frame_rotations[joint] = joint_rs

        # Update dictionary with current angles
        for joint in keypoints['joints']:
            if joint not in exclude_joints_from_output:
                keypoints[joint + '_angles'].append(frame_rotations.get(joint, np.array([0., 0., 0.])))

    # Convert joint angles list to numpy arrays
    for joint in keypoints['joints']:
        if joint not in exclude_joints_from_output:
            keypoints[joint + '_angles'] = np.array(keypoints[joint + '_angles'])

    #print(f"Frame {framenum}, Left Hip Angles (rad): {frame_rotations['left_hip']}")
    #print(f"Frame {framenum}, Right Hip Angles (rad): {frame_rotations['right_hip']}")

    return keypoints

def process_chunk(keypoints_chunk, num_keypoints):
    # Apply rotation, filtering, etc., on the chunk
    R = utils.get_R_z(np.pi / 2)
    for kpt_num in range(num_keypoints):
        keypoints_chunk[kpt_num] = R @ keypoints_chunk[kpt_num]

    keypoints_dict = convert_to_dictionary(keypoints_chunk)
    add_hips_and_neck(keypoints_dict)
    #keypoints_dict = median_filter(keypoints_dict)
    get_bone_lengths(keypoints_dict)
    get_base_skeleton(keypoints_dict)
    angles_chunk = calculate_joint_angles(keypoints_dict)

    # Exclude unwanted joints from the angles dictionary
    exclude_joints_from_output = ['hips', 'neck', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']
    angles_dict = {}
    for joint in angles_chunk['joints']:
        if joint not in exclude_joints_from_output:
            angles_dict[joint + '_angles'] = angles_chunk[joint + '_angles']

    # Memory management
    del keypoints_dict, angles_chunk
    gc.collect()

    return angles_dict


def calculate_angles_tpose(keypoints, chunk_size=2500):
    combined_angles = {}
    num_frames = keypoints.shape[0]
    num_keypoints = keypoints.shape[2]

    # Transpose to (keypoints, 3, frames) for processing
    keypoints = np.transpose(keypoints, (2, 1, 0))  # (keypoints, xyz, frames)

    # Prepare inputs for multiprocessing
    chunks = [(keypoints[:, :, start:min(start + chunk_size, num_frames)], num_keypoints)
              for start in range(0, num_frames, chunk_size)]

    # Use multiprocessing to process each chunk in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        all_angles = pool.starmap(process_chunk, chunks)

    # Combine all angles into a single dictionary
    for d in all_angles:
        for key, value in d.items():
            if key in combined_angles:
                combined_angles[key] = np.concatenate((combined_angles[key], value), axis=0)
            else:
                combined_angles[key] = value

    return combined_angles


def calculate_angular_speed_error(gt, pred, sample_rate, smoothing_window=51, polyorder=3, post_smoothing = False):
    """
    Calculate the angular speed for each joint over all frames.

    :param keypoints: Dictionary containing joint angles as numpy arrays.
    :param frame_time_interval: Time interval between consecutive frames (in seconds).
    :return: Updated keypoints dictionary with angular speeds for each joint.
    """
    angular_speed_dict = {}
    angular_speed = {}

    for joint in gt:
        # Retrieve ground truth and predicted angles
        angles_gt = gt[joint]
        angles_pred = pred[joint]

        assert angles_gt.shape[1] == angles_pred.shape[1], "The number of frames must match."

        # Initialize smoothed prediction array with the same shape
        angles_pred_smoothed = np.zeros_like(angles_pred)
        angles_gt_smoothed = np.zeros_like(angles_gt)

        # Apply smoothing along each axis
        for axis in range(angles_pred.shape[1]):
            if angles_pred.shape[0] < smoothing_window:
                angles_pred_smoothed[:, axis] = angles_pred[:, axis]
                angles_gt_smoothed[:, axis] = angles_gt[:, axis]
            else:
                angles_pred_smoothed[:, axis] = savgol_filter(angles_pred[:, axis], smoothing_window, polyorder)
                angles_gt_smoothed[:, axis] = savgol_filter(angles_gt[:, axis], smoothing_window, polyorder)

        # Calculate angular velocities (central difference)
        #CHECK AXIS!
        velocity_predicted = np.gradient(angles_pred_smoothed, 1 / sample_rate, axis=0)
        velocity_target = np.gradient(angles_gt_smoothed, 1 / sample_rate, axis=0)

        # Smooth velocity profiles
        if post_smoothing and velocity_predicted.shape[0] >= smoothing_window:
            for axis in range(velocity_predicted.shape[1]):
                velocity_predicted[:, axis] = savgol_filter(velocity_predicted[:, axis], smoothing_window, polyorder)
                velocity_target[:, axis] = savgol_filter(velocity_target[:, axis], smoothing_window, polyorder)

        # Compute MPJAVE
        mpjave = np.degrees(np.abs(velocity_target - velocity_predicted))

        angular_speed[joint] = {
            "mpjave": mpjave  # Storing the full array for aggregation outside the loop
        }

    # Aggregate MPJAVE across all joints
    all_mpjave = np.concatenate([angular_speed[joint]["mpjave"].flatten() for joint in angular_speed])

    # Compute overall statistical summaries
    overall_metrics = {
        "mean": np.mean(all_mpjave),
        "std": np.std(all_mpjave),
        "median": np.percentile(all_mpjave, 50),
        "Q1": np.percentile(all_mpjave, 25),
        "Q3": np.percentile(all_mpjave, 75),
        "IQR": np.percentile(all_mpjave, 75) - np.percentile(all_mpjave, 25),
        "loval": np.percentile(all_mpjave, 25) - 1.5 * (
                    np.percentile(all_mpjave, 75) - np.percentile(all_mpjave, 25)),
        "hival": np.percentile(all_mpjave, 75) + 1.5 * (
                    np.percentile(all_mpjave, 75) - np.percentile(all_mpjave, 25)),
    }

    # Compute whisker values for filtering outliers
    wiskhi = all_mpjave[all_mpjave <= overall_metrics["hival"]]
    wisklo = all_mpjave[all_mpjave >= overall_metrics["loval"]]
    overall_metrics["actual_hival"] = np.max(wiskhi) if len(wiskhi) > 0 else np.nan
    overall_metrics["actual_loval"] = np.min(wisklo) if len(wisklo) > 0 else np.nan

    # Add overall metrics to the result
    angular_speed_dict = overall_metrics

    return angular_speed_dict
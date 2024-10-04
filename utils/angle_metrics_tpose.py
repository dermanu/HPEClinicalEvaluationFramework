import numpy as np
from utils import angle_metrics_tpose_utils as utils
from scipy.signal import medfilt
import gc
import multiprocessing


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
    #this defines a generic skeleton to which we can apply rotations to
    body_lengths = kpts['bone_lengths']
    normalization = kpts['bone_lengths'][normalization_bone]

    directions = {
        'left_hip': np.array([1, 0.2, 0]),  # Adjusted direction to avoid symmetry
        'left_knee': np.array([0, -1, 0]), 'left_ankle': np.array([0, -1, 0]),
        'right_hip': np.array([-1, -0.2, 0]),  # Adjusted direction to avoid symmetry
        'right_knee': np.array([0, -1, 0]), 'right_ankle': np.array([0, -1, 0]),
        'neck': np.array([0, 1, 0]),
        'left_shoulder': np.array([1, 0, 0]), 'left_elbow': np.array([1, 0, 0]), 'left_wrist': np.array([1, 0, 0]),
        'right_shoulder': np.array([-1, 0, 0]), 'right_elbow': np.array([-1, 0, 0]),
        'right_wrist': np.array([-1, 0, 0]),
        'left_heel': np.array([0, -1, 0]), 'right_heel': np.array([0, -1, 0]),
        'left_foot_index': np.array([0.5, -0.5, 0]), 'right_foot_index': np.array([-0.5, -0.5, 0])
    }

    #base skeleton set by multiplying offset directions by measured bone lengths. In this case we use the average of two sided limbs. E.g left and right hip averaged
    base_skeleton = {'hips': np.array([0, 0, 0])}

    for joint_type in ['hip', 'knee', 'ankle', 'shoulder', 'elbow', 'wrist']:
        base_skeleton['left_' + joint_type] = directions['left_' + joint_type] * (
                (body_lengths['left_' + joint_type] + body_lengths['right_' + joint_type]) / (2 * normalization))
        base_skeleton['right_' + joint_type] = directions['right_' + joint_type] * (
                (body_lengths['left_' + joint_type] + body_lengths['right_' + joint_type]) / (2 * normalization))

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
    exclude_joints_from_output = ['neck', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']

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

    return keypoints

def process_chunk(keypoints_chunk, num_keypoints):
    # Apply rotation, filtering, etc., on the chunk
    R = utils.get_R_z(np.pi / 2)
    for kpt_num in range(num_keypoints):
        keypoints_chunk[kpt_num] = R @ keypoints_chunk[kpt_num]

    keypoints_dict = convert_to_dictionary(keypoints_chunk)
    add_hips_and_neck(keypoints_dict)
    filtered_keypoints = median_filter(keypoints_dict)
    get_bone_lengths(filtered_keypoints)
    get_base_skeleton(filtered_keypoints)
    angles_chunk = calculate_joint_angles(filtered_keypoints)

    # Exclude unwanted joints from the angles dictionary
    exclude_joints_from_output = ['neck', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']
    angles_dict = {}
    for joint in angles_chunk['joints']:
        if joint not in exclude_joints_from_output:
            angles_dict[joint + '_angles'] = angles_chunk[joint + '_angles']

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
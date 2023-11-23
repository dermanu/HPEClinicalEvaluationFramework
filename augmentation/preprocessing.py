import torch
import numpy as np

def center2pelvis(keypoints_2d, pelvis_i):
    """ pelvis -> (0, 0) """

    n_joints = keypoints_2d.shape[2]
    pelvis_point = keypoints_2d[:, :, pelvis_i, :]

    return keypoints_2d - pelvis_point.unsqueeze(2).repeat(1, 1, n_joints, 1)  # in each view: joint coords - pelvis coords


def dist2pelvis(keypoints_2d_in_view, pelvis_i):
    return torch.mean(torch.cat([
        torch.norm(
            keypoints_2d_in_view[i] -\
            keypoints_2d_in_view[pelvis_i]
        ).unsqueeze(0)
        for i in range(keypoints_2d_in_view.shape[0])
        if i != pelvis_i
    ])).unsqueeze(0)


def normalize_keypoints(keypoints_2d, pelvis_center_kps, normalization, pelvis_i):
    """ "divided by its Frobenius norm in the preprocessing" """

    batch_size, n_views = keypoints_2d.shape[0], keypoints_2d.shape[1]

    if pelvis_center_kps:
        kps = center2pelvis(keypoints_2d, pelvis_i)
    else:
        kps = keypoints_2d

    if normalization == 'd2pelvis':
        scaling = torch.cat([
            torch.max(
                torch.cat([
                    dist2pelvis(kps[batch_i, view_i])
                    for view_i in range(n_views)
                ]).unsqueeze(0)
            ).unsqueeze(0).repeat(1, n_views)  # same for each view
            for batch_i in range(batch_size)
        ])
    elif normalization == 'fro':
        scaling = torch.cat([
            torch.cat([
                torch.norm(kps[batch_i, view_i], p='fro').unsqueeze(0)
                for view_i in range(n_views)
            ]).unsqueeze(0)
            for batch_i in range(batch_size)
        ])
    elif normalization == 'maxfro':
        scaling = torch.cat([
            torch.max(
                torch.cat([
                    torch.norm(kps[batch_i, view_i], p='fro').unsqueeze(0)
                    for view_i in range(n_views)
                ]).unsqueeze(0)
            ).unsqueeze(0).repeat(1, n_views)  # same for each view
            for batch_i in range(batch_size)
        ])
    elif normalization == 'fixed':
        factor = 40.0  # todo to be scaled with K
        scaling = factor * torch.ones(batch_size, n_views)

    return torch.cat([
        torch.cat([
            (
                kps[batch_i, view_i] / scaling[batch_i, view_i]
            ).unsqueeze(0)
            for view_i in range(n_views)
        ]).unsqueeze(0)
        for batch_i in range(batch_size)
    ])

@staticmethod
def _reparameterize_pelvis_in_origin(kps, pelvis_i):
    pelvis_in_world = kps[pelvis_i].reshape(3, 1)

    origin = np.zeros_like(pelvis_in_world)
    t_from_pelvis2origin = origin - pelvis_in_world

    return np.float64([
        kp.reshape(3, 1) + t_from_pelvis2origin.reshape(3, 1)
        for kp in kps
    ]).squeeze(-1)


def _preprocess(self):
    self.labels['table']['keypoints'] = np.float64([
        self._reparameterize_pelvis_in_origin(kps, 6)
        for kps in self.labels['table']['keypoints']
    ])



def preprocess_extrinsics(self, image, shot, camera_idx, retval_camera):
    if False:  # using GTs ... self.crop:
        image = crop_image(image, bbox)
        bbox = self.get_bbox(shot, camera_idx)
        retval_camera.update_after_crop(bbox)

    if False:  # using GTs ... self.resample_same_K:
        # todo move to __init__
        square = (0, 0, 1000, 1000)  # get rid of 1K + eps
        image = crop_image(image, square)
        retval_camera.update_after_crop(square)

        # have same intrinsics
        image = resample_image(
            image, TARGET_INTRINSICS, retval_camera.K
        )
        retval_camera.K = self.target_K()

    if self.look_at_pelvis:
        pelvis_index = 6  # H36M dataset, not CMU

        pelvis_vector = retval_camera.world2cam()(
            shot['keypoints'][:self.num_keypoints]  # in world
        )[pelvis_index]

        # find rotation matrix to align pelvis to z ...
        z_axis = [0, 0, 1]
        Rt = rotation_matrix_from_vectors_rodrigues(
            pelvis_vector, z_axis
        )

        # ... "At that point, after you re-sample, camera translation should be [0, 0, d_pelvis]"
        retval_camera.update_extrinsics(Rt)

    if self.scale2m:
        scaling = 1e3  # mm -> m
        retval_camera.scale_extrinsics(scaling)
        retval_camera.scale_K(np.sqrt(scaling))  # see https://ksimek.github.io/perspective_camera_toy.html




def skeleton_morphing(keypoints_2d_COCO):
    """Morphing between two skeletons of different datasets and therefore different joint postions"""
    model_skel_morph = torch.load('model_skeleton_morph_S1_gh.pt')
    model_skel_morph.eval()
    keypoints_2d_HM36M = model_skel_morph(keypoints_2d_COCO)

    return keypoints_2d_HM36M
import time

import numpy as np
import torch
from tqdm import tqdm

from detector.apis import get_detector
from trackers import track
from alphapose.models import builder
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.vis import getTime

# Load pose model
pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

def process_video_alphaPose(input_video_path, cfg_path, checkpoint_path):
    det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode=mode,
                                 queueSize=args.qsize)
    det_worker = det_loader.start()

    pose_model.to(args.device)
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    data_len = det_loader.length
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    batchSize = args.posebatch
    try:
        for i in im_names_desc:
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                if orig_img is None:
                    break
                # Pose Estimation
                inps = inps.to(args.device)
                img_center = torch.Tensor((orig_img.shape[1], orig_img.shape[0])).float().to(args.device) / 2
                img_center = img_center.unsqueeze(0).repeat(inps.shape[0], 1)

                pose_output = pose_model(
                    inps, flip_test=args.flip,
                    bboxes=cropped_boxes.to(args.device),
                    img_center=img_center
                )

                smpl_output = {
                    'pred_uvd_jts': pose_output.pred_uvd_jts.cpu()[new_ids],
                    'maxvals': pose_output.maxvals.cpu()[new_ids],
                    'transl': pose_output.transl.cpu()[new_ids],
                    'pred_vertices': pose_output.pred_vertices.cpu()[new_ids],
                    'pred_xyz_jts_24': pose_output.pred_xyz_jts_24_struct.cpu()[new_ids] * 2,  # convert to meters
                    'smpl_faces': torch.from_numpy(pose_model.smpl.faces.astype(np.int32))
                }



# Now, `keypoints_data` contains the parsed keypoints from the AlphaPose output file

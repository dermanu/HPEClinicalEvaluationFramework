"""
One-time extractor: dump per-participant G + aligned best-camera HPE arrays
from the existing .pth morph datasets.

Run from the repo root, ideally as a module so the package imports resolve:
    python -m skeletonMorphing.extractGTHPE --data_path . --out synth_arrays
"""
import os, sys, types

# --- make the repo importable and stub inference-only deps so unpickling works ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# readDatasetMorph imports models.mediapipeMono + cv2 at module load, but the
# extractor never runs inference -> provide harmless stubs if they're absent.
_models = types.ModuleType("models")
_mp = types.ModuleType("models.mediapipeMono")
_models.mediapipeMono = _mp
sys.modules.setdefault("models", _models)
sys.modules.setdefault("models.mediapipeMono", _mp)
try:
    import cv2  # noqa: F401
except Exception:
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))

import argparse
import numpy as np
import torch
import skeletonMorphing.readDatasetMorph  # noqa: F401  registers classes for the unpickler


def best_camera_pose(pose_inf, conf_inf):
    best = conf_inf.mean(axis=2).argmax(axis=1)          # (F,)
    idx = np.arange(pose_inf.shape[0])
    return pose_inf[idx, best]                            # (F,16,3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="folder containing morph_dataset/")
    ap.add_argument("--out", required=True, help="output folder for per-participant .npz")
    ap.add_argument("--suffix", default="_mediapipe_dataset.pth",
                    help="filename suffix of your .pth files (adjust if different)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    morph_dir = os.path.join(args.data_path, "morph_dataset")
    files = [f for f in os.listdir(morph_dir) if f.endswith(args.suffix)]
    print(f"found {len(files)} participant datasets in {morph_dir}")
    if not files:
        print("  -> none matched. Check --data_path and --suffix vs your real filenames.")
        return

    for f in sorted(files):
        stem = f.split(args.suffix)[0]
        par = stem.replace("par_", "").replace("par", "").split("_")[0]
        rdf = torch.load(os.path.join(morph_dir, f), map_location="cpu", weights_only=False)

        G_list, HPE_list = [], []
        for d in rdf.datasets:
            if getattr(d, "csv_data", None) is None or np.size(d.csv_data) == 0:
                continue
            d.align_procrustes()
            G = np.asarray(d.csv_data, dtype=np.float64)
            HPE = best_camera_pose(np.asarray(d.pose_inf, dtype=np.float64),
                                   np.asarray(d.confidences_inf, dtype=np.float64))
            ok = ~(np.isnan(G).any(axis=(1, 2)) | np.isnan(HPE).any(axis=(1, 2)))
            G_list.append(G[ok]); HPE_list.append(HPE[ok])

        if not G_list:
            print(f"  par {par}: no usable frames, skipped"); continue
        G = np.concatenate(G_list); HPE = np.concatenate(HPE_list)
        n = len(G); cut = int(0.8 * n)
        np.savez_compressed(os.path.join(args.out, f"par_{par}.npz"),
                            G=G, HPE=HPE, train_idx=np.arange(cut), test_idx=np.arange(cut, n))
        print(f"  par {par}: {n} frames -> par_{par}.npz  "
              f"(GT mm range {np.nanmin(G):.0f}..{np.nanmax(G):.0f})")
    print("done.")


if __name__ == "__main__":
    main()
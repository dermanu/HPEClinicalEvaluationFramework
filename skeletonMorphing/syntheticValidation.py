"""
Synthetic-deviation validation of the Skeleton Morphing Module.

Purpose (answers reviewers DE/ES Major 4 and the paper's own stated future work):
    Stress-test the trained morphing model against *known* anatomical deviations to
    show it (a) removes a controlled systematic joint-centre offset and (b) does NOT
    absorb/smooth random tracking noise (i.e. it harmonises without masking error).

Method (paired clean/noisy streams = the trick that makes both questions measurable):
    From real Plug-in-Gait ground-truth skeletons G (N, 16, 3):
        S_clean = G + systematic(G)            # known offset morphing SHOULD remove
        S_noisy = S_clean + noise              # known noise morphing MUST preserve
    Run the trained morpher M on both:
        recovery  : how much of the known systematic offset M removes (clean stream)
        noise-ratio: std(M(S_noisy) - M(S_clean)) / std(noise)   ~1 honest, <1 masking

This script is dependency-light. With --demo it fabricates data and uses a NumPy stub
"morpher" so the whole pipeline (and every metric) runs without torch or private data.
For the real run, point --gt at an (N,16,3) or (N,48) .npy of GT skeletons and --model
at model_skeleton_morph_mediapipe_final.pth.

It also writes all_ground_truths.npy / all_hpe_truths.npy / all_predictions.npy in the
exact (N,48) layout statistics/morphing_statistics.py expects, so you can pipe results
straight into your existing stats script.
"""

import argparse
import os
import numpy as np

# ----------------------------------------------------------------------------- #
# Joint order (must match trainFinalSkeletonMorphing.py). Index 0..15.           #
# ----------------------------------------------------------------------------- #
JOINT_NAMES = [
    'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow',
    'right_wrist', 'left_wrist', 'right_hip', 'left_hip',
    'right_knee', 'left_knee', 'right_ankle', 'left_ankle',
    'right_heel', 'left_heel', 'right_foot_index', 'left_foot_index',
]
N_JOINTS = len(JOINT_NAMES)

# Distal joints get larger offsets/noise (mirrors the real wrist/elbow finding).
# weight ~ how far down the kinematic chain a joint sits.
DISTALITY = np.array([
    0.5, 0.5,   # shoulders (proximal)
    0.8, 0.8,   # elbows
    1.0, 1.0,   # wrists (most distal upper)
    0.4, 0.4,   # hips (proximal)
    0.7, 0.7,   # knees
    0.9, 0.9,   # ankles
    1.0, 1.0,   # heels
    1.0, 1.0,   # foot index (most distal lower)
])


# ----------------------------------------------------------------------------- #
# I/O helpers                                                                    #
# ----------------------------------------------------------------------------- #
def to_njoints3(arr):
    """Accept (N,48) or (N,16,3); return (N,16,3) float64."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == N_JOINTS * 3:
        arr = arr.reshape(-1, N_JOINTS, 3)
    assert arr.ndim == 3 and arr.shape[1:] == (N_JOINTS, 3), \
        f"expected (N,16,3) or (N,48), got {arr.shape}"
    return arr


def flatten48(arr):
    return arr.reshape(arr.shape[0], N_JOINTS * 3)


# ----------------------------------------------------------------------------- #
# Deviation model: the synthetic "HPE-like" source                              #
# ----------------------------------------------------------------------------- #
# --- kinematic chain for the anatomical model (indices match JOINT_NAMES) -------
# parent used to define each joint's bone segment (joint - parent).
_PARENT = {0: 6, 1: 7, 2: 0, 3: 1, 4: 2, 5: 3, 6: 0, 7: 1,
           8: 6, 9: 7, 10: 8, 11: 9, 12: 10, 13: 11, 14: 10, 15: 11}
# flexion angle at joint j = angle(a->j , c->j). Endpoints borrow a neighbour's angle.
_ANGLE = {0: (6, 0, 2), 1: (7, 1, 3), 2: (0, 2, 4), 3: (1, 3, 5),
          4: (0, 2, 4), 5: (1, 3, 5), 6: (0, 6, 8), 7: (1, 7, 9),
          8: (6, 8, 10), 9: (7, 9, 11), 10: (8, 10, 14), 11: (9, 11, 15),
          12: (8, 10, 14), 13: (9, 11, 15), 14: (8, 10, 14), 15: (9, 11, 15)}
_HIPS = (6, 7)


def _unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.where(n < 1e-9, 1.0, n)


def _radial_dirs(ref):
    """
    ref (16,3) reference (mean) pose -> (16,3) outward, perpendicular-to-bone unit
    directions: the skin-surface direction a surface keypoint sits along relative to
    the internal joint centre. Frozen across frames => the LINEAR offset stays a
    constant translation (affine-recoverable).
    """
    center = ref[list(_HIPS)].mean(axis=0)             # pelvis ~ body centre
    dirs = np.zeros((N_JOINTS, 3))
    for j in range(N_JOINTS):
        seg = _unit(ref[j] - ref[_PARENT[j]])          # bone direction into joint
        out = ref[j] - center                          # outward from body centre
        radial = out - (out @ seg) * seg               # component perpendicular to bone
        if np.linalg.norm(radial) < 1e-6:              # degenerate -> any perp to seg
            tmp = np.array([1.0, 0, 0]) if abs(seg[0]) < 0.9 else np.array([0, 1.0, 0])
            radial = tmp - (tmp @ seg) * seg
        dirs[j] = _unit(radial)
    return dirs


def _flexion(G):
    """(N,16) flexion in [0,1]: 0 = straight, ~1 = fully bent, per joint per frame."""
    N = G.shape[0]
    flex = np.zeros((N, N_JOINTS))
    for j in range(N_JOINTS):
        a, jj, c = _ANGLE[j]
        u = _unit(G[:, a] - G[:, jj]); v = _unit(G[:, c] - G[:, jj])
        cos = np.clip((u * v).sum(axis=1), -1.0, 1.0)
        theta = np.arccos(cos)                          # 0=folded, pi=straight
        flex[:, j] = (np.pi - theta) / np.pi            # 0=straight, 1=folded
    return flex


def make_systematic(G, lin_mm, nonlin_mm, seed=0, model="anatomical", ref_pose=None):
    """
    Deterministic, *known* systematic offset (N,16,3) = linear + nonlinear.

    model="anatomical" (default, recommended):
        linear    = constant outward radial (segment-normal) offset, frozen per joint
                    -> models the stable surface-keypoint vs internal-joint-centre gap;
                       affine-recoverable.
        nonlinear = same radial direction, magnitude modulated by JOINT FLEXION
                    (de-meaned so it is purely pose-varying) -> models pose-dependent
                    soft-tissue artefact (cf. Benoit et al.); NOT a function of the
                    joint's own xyz, so a per-joint affine baseline cannot capture it.

    model="generic":
        the original random-direction constant + sine-of-position warp (kept for
        comparison / ablation).

    lin_mm, nonlin_mm scale the linear/nonlinear magnitudes (each * DISTALITY[j]).
    seed only affects the "generic" model (anatomical is deterministic from geometry).
    ref_pose : (16,3) SHARED canonical pose for the radial directions. Pass a single
               global mean pose across participants so the linear offset direction is
               an anatomical constant (affine-recoverable + generalises). Defaults to
               this array's own mean pose (fine for a single dataset).
    """
    N = G.shape[0]

    if model == "anatomical":
        ref = G.mean(axis=0) if ref_pose is None else np.asarray(ref_pose, float)
        dirs = _radial_dirs(ref)                        # (16,3) shared frozen directions
        linear = np.broadcast_to(dirs * (lin_mm * DISTALITY)[:, None], (N, N_JOINTS, 3))

        # nonlinear: radial offset whose MAGNITUDE follows joint flexion. Flexion is a
        # nonlinear function of the parent/child joints, so a per-joint affine in the
        # joint's OWN coordinates cannot reproduce it -> only a context-aware morpher can.
        # De-meaned so it is PURELY pose-varying (a linear map's best constant fit gains
        # nothing -> the baseline genuinely fails here), then normalised to ~unit std so
        # nonlin_mm sets the ACTUAL amplitude. (Raw de-meaned flexion has std ~0.1, which
        # silently shrank the offset ~10x and pushed it into the noise floor.)
        flex = _flexion(G)                              # (N,16) in [0,1]
        flex = flex - flex.mean(axis=0, keepdims=True)  # purely pose-varying
        flex = flex / (flex.std(axis=0, keepdims=True) + 1e-6)   # unit std
        flex = np.clip(flex, -3.0, 3.0)                 # tame rare outliers
        nonlinear = (dirs[None] * (nonlin_mm * DISTALITY)[None, :, None]
                     * flex[:, :, None])                # (N,16,3), substantial + pose-varying
        return linear + nonlinear

    elif model == "generic":
        rng = np.random.default_rng(seed)
        dirs = _unit(rng.normal(size=(N_JOINTS, 3)))
        linear = np.broadcast_to(dirs * (lin_mm * DISTALITY)[:, None], (N, N_JOINTS, 3))
        span = (G.max(axis=0) - G.min(axis=0)); span[span == 0] = 1.0
        Gn = 2.0 * (G - G.min(axis=0)) / span - 1.0
        phase = rng.uniform(0, 2 * np.pi, size=(N_JOINTS, 3))
        freq = rng.uniform(1.0, 2.0, size=(N_JOINTS, 3))
        warp = np.sin(freq[None] * Gn * np.pi + phase[None])
        nonlinear = warp * (nonlin_mm * DISTALITY)[:, None]
        return linear + nonlinear

    raise ValueError(f"unknown model: {model}")


def per_frame_offset_mag(delta):
    """(N,16,3) residual field -> (16,) mean over frames of per-joint offset magnitude.
    Use on the CLEAN (noise-free) stream so it captures constant AND pose-varying offset."""
    return np.linalg.norm(delta, axis=2).mean(axis=0)


def make_noise(G, sigma_mm, seed=1):
    """Independent zero-mean Gaussian tracking noise (N,16,3), distal-heavier."""
    rng = np.random.default_rng(seed)
    sigma = (sigma_mm * DISTALITY)[None, :, None]      # (1,16,1)
    return rng.normal(scale=1.0, size=G.shape) * sigma


# ----------------------------------------------------------------------------- #
# Morpher wrappers                                                               #
# ----------------------------------------------------------------------------- #
def load_torch_morpher(model_path):
    """Return fn: (M,16,3)->(M,16,3) using the trained Synthesizer."""
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval().to(device)

    def morph(x_njoints3):
        with torch.no_grad():
            x = torch.as_tensor(flatten48(x_njoints3), dtype=torch.float32, device=device)
            y = model(x).cpu().numpy()
        return to_njoints3(y)
    return morph


def make_demo_morpher(G_ref, recovery=0.9, noise_keep=0.6):
    """
    NumPy stand-in for --demo ONLY (the real run uses the trained Synthesizer,
    which does NOT know G). Emulates a realistic trained morpher by decomposing
    the source's deviation from the (known, demo-only) ground truth into a
    systematic part (frame-mean) and a fluctuation part, then removing `recovery`
    of the systematic while keeping only `noise_keep` of the fluctuation. The
    latter is the masking we WANT the noise-ratio metric to expose (ratio<1).
    Closed over G_ref only, so it works unchanged across the magnitude sweep.
    """
    def morph(x):
        delta = x - G_ref
        sys = delta.mean(axis=0, keepdims=True)     # systematic (noise ~cancels)
        fluct = delta - sys                         # ~ noise + nonlinear residual
        return G_ref + (1 - recovery) * sys + noise_keep * fluct
    return morph


# ----------------------------------------------------------------------------- #
# Metrics: the systematic-recovery / noise-preservation decomposition           #
# ----------------------------------------------------------------------------- #
def systematic_vectors(delta):
    """Per-joint frame-mean of a residual field (N,16,3) -> (16,3). Noise cancels."""
    return delta.mean(axis=0)


def per_joint_norm(vec16x3):
    return np.linalg.norm(vec16x3, axis=1)            # (16,)


def evaluate(G, S_clean, S_noisy, morph):
    """Run morpher on both streams and compute the decomposition metrics."""
    M_clean = morph(S_clean)
    M_noisy = morph(S_noisy)

    # --- systematic recovery (use the noise-free stream) -----------------------
    # per-frame offset magnitude: captures constant AND pose-varying (flexion) offset
    sys_before = per_frame_offset_mag(S_clean - G)                # injected (16,)
    sys_after = per_frame_offset_mag(M_clean - G)                 # residual (16,)
    recovery = 1.0 - np.divide(sys_after, sys_before,
                               out=np.zeros_like(sys_after),
                               where=sys_before > 1e-9)

    # --- noise preservation (clean-vs-noisy isolates the known noise) ----------
    injected_noise = S_noisy - S_clean                            # = noise (known)
    propagated_noise = M_noisy - M_clean                          # noise through M
    # per-joint std over (frames, xyz)
    std_in = injected_noise.std(axis=(0, 2))                      # (16,)
    std_out = propagated_noise.std(axis=(0, 2))                   # (16,)
    noise_ratio = np.divide(std_out, std_in,
                            out=np.ones_like(std_out),
                            where=std_in > 1e-9)

    # --- residual directional bias (after morphing, clean stream) --------------
    bias_vec = systematic_vectors(M_clean - G)                    # (16,3)

    return dict(
        sys_before=sys_before, sys_after=sys_after, recovery=recovery,
        noise_ratio=noise_ratio, bias_vec=bias_vec,
        M_clean=M_clean, M_noisy=M_noisy,
    )


def linear_baseline(G, S, fit_frac=0.5):
    """
    Per-joint best-fit affine map S->G (least squares) on a fit split, applied to
    the held-out split. Returns morphed (N_test,16,3) and the test indices.
    Shows a LINEAR correction cannot undo the nonlinear warp (Rapczynski-style).
    """
    N = G.shape[0]
    cut = int(N * fit_frac)
    out = np.empty((N - cut, N_JOINTS, 3))
    for j in range(N_JOINTS):
        X = S[:cut, j, :]
        Y = G[:cut, j, :]
        Xa = np.hstack([X, np.ones((cut, 1))])            # augment for translation
        W, *_ = np.linalg.lstsq(Xa, Y, rcond=None)        # (4,3)
        Xt = np.hstack([S[cut:, j, :], np.ones((N - cut, 1))])
        out[:, j, :] = Xt @ W
    return out, np.arange(cut, N)


# ----------------------------------------------------------------------------- #
# Reporting                                                                      #
# ----------------------------------------------------------------------------- #
def print_table(res, title):
    print(f"\n=== {title} ===")
    print(f"{'joint':<18}{'sys_before':>11}{'sys_after':>11}{'recovery%':>11}{'noise_ratio':>13}")
    for j, name in enumerate(JOINT_NAMES):
        print(f"{name:<18}{res['sys_before'][j]:>11.1f}{res['sys_after'][j]:>11.1f}"
              f"{100*res['recovery'][j]:>11.1f}{res['noise_ratio'][j]:>13.3f}")
    print("-" * 64)
    print(f"{'OVERALL (mean)':<18}{res['sys_before'].mean():>11.1f}"
          f"{res['sys_after'].mean():>11.1f}{100*res['recovery'].mean():>11.1f}"
          f"{res['noise_ratio'].mean():>13.3f}")


def sweep_figure(G, sweep_mm, sigma_mm, nonlin_mm, morph, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resid, ratio = [], []
    for mag in sweep_mm:
        syst = make_systematic(G, lin_mm=mag, nonlin_mm=nonlin_mm, seed=int(mag))
        noise = make_noise(G, sigma_mm=sigma_mm, seed=99)
        S_clean = G + syst
        S_noisy = S_clean + noise
        r = evaluate(G, S_clean, S_noisy, morph)
        resid.append(r['sys_after'].mean())
        ratio.append(r['noise_ratio'].mean())

    fig, ax1 = plt.subplots(figsize=(6.2, 4.2))
    c1 = ax1.plot(sweep_mm, resid, 'o-', color='#c0392b', label='residual systematic offset (mm)')
    ax1.set_xlabel("injected systematic offset magnitude (mm)")
    ax1.set_ylabel("residual systematic after morphing (mm)", color='#c0392b')
    ax1.tick_params(axis='y', labelcolor='#c0392b')
    ax1.axhline(0, color='#c0392b', lw=0.6, ls=':')

    ax2 = ax1.twinx()
    c2 = ax2.plot(sweep_mm, ratio, 's--', color='#2c3e50', label='noise-preservation ratio')
    ax2.axhline(1.0, color='#2c3e50', lw=0.6, ls=':')
    ax2.set_ylabel("noise-preservation ratio (1 = honest)", color='#2c3e50')
    ax2.tick_params(axis='y', labelcolor='#2c3e50')
    ax2.set_ylim(0, 1.3)

    lines = c1 + c2
    ax1.legend(lines, [l.get_label() for l in lines], loc='center right', fontsize=8)
    ax1.set_title("Morphing removes known offset, should preserve known noise")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\n[figure] wrote {out_png}")


# ----------------------------------------------------------------------------- #
# Main                                                                           #
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Synthetic-deviation validation of the morphing model")
    ap.add_argument("--gt", type=str, default=None,
                    help="(N,16,3) or (N,48) .npy of Plug-in-Gait ground-truth skeletons")
    ap.add_argument("--model", type=str, default="models/trained/model_skeleton_morph_mediapipe_final.pth")
    ap.add_argument("--demo", action="store_true", help="fabricate data + use NumPy stub morpher (no torch)")
    ap.add_argument("--lin_mm", type=float, default=40.0, help="linear systematic offset magnitude (mm)")
    ap.add_argument("--nonlin_mm", type=float, default=15.0, help="nonlinear warp amplitude (mm)")
    ap.add_argument("--sigma_mm", type=float, default=6.0, help="random tracking-noise std (mm)")
    ap.add_argument("--n_demo", type=int, default=4000, help="frames to fabricate in --demo")
    ap.add_argument("--outdir", type=str, default="skeletonMorphing/synthetic_validation_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- data ----------------------------------------------------------------- #
    if args.demo:
        rng = np.random.default_rng(0)
        # Fabricate plausible standing skeletons: a base pose + per-frame sway.
        base = rng.normal(scale=200, size=(N_JOINTS, 3))
        sway = rng.normal(scale=40, size=(args.n_demo, N_JOINTS, 3))
        G = base[None] + sway
        morph = make_demo_morpher(G, recovery=0.9, noise_keep=0.6)
        print(f"[demo] fabricated G with shape {G.shape}; using NumPy demo morpher (recovery=0.9, noise_keep=0.6)")
    else:
        assert args.gt is not None, "--gt is required unless --demo"
        G = to_njoints3(np.load(args.gt))
        morph = load_torch_morpher(args.model)
        print(f"[real] loaded G {G.shape} and morpher from {args.model}")

    # --- IMPORTANT: coordinate frame -----------------------------------------
    # The trained Synthesizer consumes poses in the SAME normalisation used in
    # training (per-sample Procrustes alignment in load_dataset_par -> align_procrustes).
    # Apply the synthetic deviation in that same space. If your GT .npy is already
    # in the model's input frame, nothing to do. Otherwise insert the matching
    # alignment here BEFORE generating S_clean/S_noisy. (Flagged: verify once.)

    # --- build the two synthetic streams -------------------------------------- #
    syst = make_systematic(G, lin_mm=args.lin_mm, nonlin_mm=args.nonlin_mm, seed=42)
    noise = make_noise(G, sigma_mm=args.sigma_mm, seed=43)
    S_clean = G + syst
    S_noisy = S_clean + noise

    # --- main evaluation ------------------------------------------------------ #
    res = evaluate(G, S_clean, S_noisy, morph)
    print_table(res, f"Morphing model | lin={args.lin_mm}mm nonlin={args.nonlin_mm}mm noise={args.sigma_mm}mm")

    # --- linear-baseline comparison on the SAME nonlinear offset -------------- #
    lin_pred, test_idx = linear_baseline(G, S_clean, fit_frac=0.5)
    lin_resid = per_joint_norm(systematic_vectors(lin_pred - G[test_idx])).mean()
    morph_resid_test = per_joint_norm(systematic_vectors(res['M_clean'][test_idx] - G[test_idx])).mean()
    print(f"\n[linear baseline] residual systematic (mm): linear={lin_resid:.1f}  "
          f"morphing={morph_resid_test:.1f}  "
          f"(morphing should be lower when nonlin_mm>0)")

    # --- sweep figure --------------------------------------------------------- #
    sweep_figure(G, sweep_mm=[10, 20, 40, 60, 80], sigma_mm=args.sigma_mm,
                 nonlin_mm=args.nonlin_mm, morph=morph,
                 out_png=os.path.join(args.outdir, "morph_sweep.png"))

    # --- save arrays for statistics/morphing_statistics.py -------------------- #
    np.save(os.path.join(args.outdir, "all_ground_truths.npy"), flatten48(G))
    np.save(os.path.join(args.outdir, "all_hpe_truths.npy"), flatten48(S_noisy))
    np.save(os.path.join(args.outdir, "all_predictions.npy"), flatten48(res['M_noisy']))
    print(f"\n[arrays] wrote all_ground_truths/all_hpe_truths/all_predictions.npy to {args.outdir}")
    print("        -> pipe into: python statistics/morphing_statistics.py "
          f"{args.outdir}/all_ground_truths.npy {args.outdir}/all_hpe_truths.npy "
          f"{args.outdir}/all_predictions.npy")


if __name__ == "__main__":
    main()
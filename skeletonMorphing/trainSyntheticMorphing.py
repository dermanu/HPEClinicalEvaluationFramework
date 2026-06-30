"""
Synthetic-offset morphing experiment (the "train a fresh model on known offsets" design).

For each VARIANT and each offset REGIME, this:
  1. builds synthetic inputs from per-participant arrays (extractGTHPE.py output):
        pure      : input = G          + systematic(G)    + noise     ; target = G
        realistic : input = alignedHPE + systematic(HPE)  + noise      ; target = G
  2. trains a FRESH Synthesizer (your architecture + recipe) on TRAIN participants,
  3. tests on HELD-OUT participants and scores:
        recovery        : fraction of the known systematic offset removed (pure: exact)
        noise-ratio     : std(M(noisy)-M(clean)) / std(noise)  -> ~1 honest, <1 masking
        linear baseline : per-joint affine; morphing should beat it when nonlinear present
        noise-free ceiling : recovery on the clean (noise-free) test stream

REGIMES: 'linear' (nonlin=0), 'nonlinear' (lin=0), 'combined'.

The split is by PARTICIPANT (mirrors your leave-n-out), so held-out anatomy is unseen.

--demo runs the entire orchestration without torch/data: it fabricates participants
and "trains" a per-joint affine via least squares, so shapes, splits, scoring, export,
and the stats hand-off are all exercised. The real run uses --arrays + torch.
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from syntheticValidation import (                      # reuse, don't duplicate
    JOINT_NAMES, N_JOINTS, to_njoints3, flatten48,
    make_systematic, make_noise, systematic_vectors, per_joint_norm,
    per_frame_offset_mag,
)

REGIMES = {
    "linear":    dict(lin=1.0, nonlin=0.0),
    "nonlinear": dict(lin=0.0, nonlin=1.0),
    "combined":  dict(lin=1.0, nonlin=1.0),
}


# --------------------------------------------------------------------------- #
# Build synthetic streams for one base array                                  #
# --------------------------------------------------------------------------- #
def build_streams(base, regime, lin_mm, nonlin_mm, sigma_mm, sys_seed, noise_seed, ref_pose=None):
    """
    base,(F,16,3) -> dict(clean, noisy, sys, noise). Offset injected in GT frame.

    sys_seed   : SHARED across participants (the systematic offset is a consistent
                 anatomical HPE-vs-PlugInGait difference; if it varied per person
                 there would be nothing generalizable to learn).
    noise_seed : per-participant/per-call (noise is irreducible, by construction).
    ref_pose   : shared canonical pose for anatomical radial directions (pass the
                 global mean pose so the linear offset direction is shared).
    """
    r = REGIMES[regime]
    sys = make_systematic(base, lin_mm=lin_mm * r["lin"], nonlin_mm=nonlin_mm * r["nonlin"],
                          seed=sys_seed, ref_pose=ref_pose)
    noise = make_noise(base, sigma_mm=sigma_mm, seed=noise_seed)
    clean = base + sys
    noisy = clean + noise
    return dict(clean=clean, noisy=noisy, sys=sys, noise=noise)


# --------------------------------------------------------------------------- #
# Trainers: torch (real) and affine (demo)                                     #
# --------------------------------------------------------------------------- #
def train_torch(X, Y, cfg):
    """X,Y: (M,16,3) inputs/targets. Returns callable morph:(N,16,3)->(N,16,3)."""
    import torch, torch.nn as nn, torch.optim as optim
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from skeletonMorphing.modelSkeletonMorphing import Synthesizer

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42); np.random.seed(42)

    # subsample every 25th (1 s) like EveryNthSampler
    Xs, Ys = X[::25], Y[::25]
    xb = torch.as_tensor(flatten48(Xs), dtype=torch.float32, device=dev)
    yb = torch.as_tensor(flatten48(Ys), dtype=torch.float32, device=dev)

    model = Synthesizer(cfg["dropout_rate"], cfg["layer_size"]).to(dev)
    crit = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

    n = xb.shape[0]; bs = cfg["batch_size"]
    for epoch in range(cfg["epochs"]):
        model.train(); perm = torch.randperm(n, device=dev); losses = []
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            loss = crit(model(xb[j]), yb[j])
            loss.backward(); opt.step(); losses.append(loss.item())
        sched.step(np.mean(losses))

    model.eval()
    def morph(a):
        import torch as T
        with T.no_grad():
            t = T.as_tensor(flatten48(a), dtype=T.float32, device=dev)
            return to_njoints3(model(t).cpu().numpy())
    return morph


def train_affine(X, Y, cfg=None):
    """Demo 'trained model': per-joint least-squares affine X->Y. Callable morpher."""
    W = []
    for j in range(N_JOINTS):
        A = np.hstack([X[:, j, :], np.ones((len(X), 1))])
        Wj, *_ = np.linalg.lstsq(A, Y[:, j, :], rcond=None)
        W.append(Wj)
    W = np.stack(W)                                     # (16,4,3)

    def morph(a):
        out = np.empty_like(a)
        for j in range(N_JOINTS):
            A = np.hstack([a[:, j, :], np.ones((len(a), 1))])
            out[:, j, :] = A @ W[j]
        return out
    return morph


# --------------------------------------------------------------------------- #
# Scoring on a held-out participant                                           #
# --------------------------------------------------------------------------- #
def score(morph, base_test, G_test, streams, variant):
    M_clean = morph(streams["clean"])
    M_noisy = morph(streams["noisy"])

    out = {}
    # noise-ratio (works for both variants; the load-bearing masking test)
    inj = streams["noisy"] - streams["clean"]                       # = noise
    prop = M_noisy - M_clean
    out["noise_ratio"] = (prop.std(axis=(0, 2)) /
                          np.where(inj.std(axis=(0, 2)) > 1e-9, inj.std(axis=(0, 2)), 1.0))

    # PA-MPJPE-style overall improvement vs target G (both variants)
    out["mpjpe_in"] = np.linalg.norm(streams["noisy"] - G_test, axis=2).mean()
    out["mpjpe_out"] = np.linalg.norm(M_noisy - G_test, axis=2).mean()

    if variant == "pure":
        # exact recovery of the known systematic (clean stream; per-frame magnitude
        # captures both the constant linear offset and the flexion-varying nonlinear one)
        sys_before = per_frame_offset_mag(streams["clean"] - G_test)
        sys_after = per_frame_offset_mag(M_clean - G_test)
        out["sys_before"] = sys_before
        out["sys_after"] = sys_after
        out["recovery"] = 1 - np.divide(sys_after, sys_before,
                                        out=np.zeros_like(sys_after), where=sys_before > 1e-9)
    return out, M_noisy


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #
def load_participants(arrays_dir):
    """Return {par_id: dict(G,HPE)} from extractGTHPE.py .npz files."""
    parts = {}
    for f in sorted(os.listdir(arrays_dir)):
        if f.startswith("par_") and f.endswith(".npz"):
            pid = f[4:-4]
            z = np.load(os.path.join(arrays_dir, f))
            parts[pid] = dict(G=z["G"], HPE=z["HPE"])
    return parts


def fabricate_participants(n_par=6, n_frames=1500, seed=0):
    rng = np.random.default_rng(seed); parts = {}
    for p in range(n_par):
        base = rng.normal(scale=200, size=(N_JOINTS, 3)) + p * 5
        G = base[None] + rng.normal(scale=40, size=(n_frames, N_JOINTS, 3))
        HPE = G + rng.normal(scale=25, size=G.shape) + rng.normal(scale=10, size=(N_JOINTS, 3))[None]
        parts[str(p)] = dict(G=G, HPE=HPE)
    return parts


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default=None, help="dir of par_*.npz from extractGTHPE.py")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--variants", nargs="+", default=["pure", "realistic"])
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES.keys()))
    ap.add_argument("--holdout", nargs="+", default=None, help="participant ids to hold out (default: last 2)")
    ap.add_argument("--lin_mm", type=float, default=40.0)
    ap.add_argument("--nonlin_mm", type=float, default=15.0)
    ap.add_argument("--sigma_mm", type=float, default=6.0)
    ap.add_argument("--outdir", default="skeletonMorphing/synthetic_experiment_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.demo:
        parts = fabricate_participants(); trainer = train_affine; cfg = None
        print(f"[demo] fabricated {len(parts)} participants; trainer = least-squares affine")
    else:
        assert args.arrays, "--arrays required unless --demo"
        parts = load_participants(args.arrays); trainer = train_torch
        cfg = dict(dropout_rate=0.2, layer_size=2048, learning_rate=5e-5,
                   weight_decay=1e-3, batch_size=64, epochs=80)   # = configFinal.yaml
        print(f"[real] loaded {len(parts)} participants; trainer = torch Synthesizer")

    pids = list(parts.keys())
    holdout = args.holdout or pids[-2:]
    train_pids = [p for p in pids if p not in holdout]
    print(f"train participants: {train_pids}\nheld-out: {holdout}\n")

    summary = []
    for variant in args.variants:
        base_key = "G" if variant == "pure" else "HPE"
        # shared canonical reference pose (radial directions must be the same for all
        # participants, else the linear offset direction is unlearnable across people)
        global_ref = np.concatenate([parts[p][base_key] for p in train_pids]).mean(axis=0)
        for regime in args.regimes:
            # --- assemble training set across train participants ---
            Xtr, Ytr = [], []
            for p in train_pids:
                base = parts[p][base_key]; G = parts[p]["G"]
                s = build_streams(base, regime, args.lin_mm, args.nonlin_mm, args.sigma_mm,
                                  sys_seed=7, noise_seed=hash((p, regime)) % 100000,
                                  ref_pose=global_ref)
                Xtr.append(s["noisy"]); Ytr.append(G)
            Xtr = np.concatenate(Xtr); Ytr = np.concatenate(Ytr)
            morph = trainer(Xtr, Ytr, cfg)
            lin_morph = train_affine(Xtr, Ytr)          # per-joint linear baseline (Rapczynski-style)

            # --- score each held-out participant ---
            nr, rec, rec_lin, mi, mo = [], [], [], [], []
            for p in holdout:
                base = parts[p][base_key]; G = parts[p]["G"]
                s = build_streams(base, regime, args.lin_mm, args.nonlin_mm, args.sigma_mm,
                                  sys_seed=7, noise_seed=hash((p, regime, "test")) % 100000,
                                  ref_pose=global_ref)
                res, M_noisy = score(morph, base, G, s, variant)
                nr.append(res["noise_ratio"].mean())
                mi.append(res["mpjpe_in"]); mo.append(res["mpjpe_out"])
                if variant == "pure":
                    rec.append(res["recovery"].mean())
                    # linear-baseline recovery on the SAME known offset (clean stream)
                    sb = per_frame_offset_mag(s["clean"] - G)
                    sa = per_frame_offset_mag(lin_morph(s["clean"]) - G)
                    rl = 1 - np.divide(sa, sb, out=np.zeros_like(sa), where=sb > 1e-9)
                    rec_lin.append(rl.mean())
                # export for morphing_statistics.py (first held-out only)
                if p == holdout[0]:
                    tag = f"{variant}_{regime}"
                    np.save(os.path.join(args.outdir, f"{tag}_all_ground_truths.npy"), flatten48(G))
                    np.save(os.path.join(args.outdir, f"{tag}_all_hpe_truths.npy"), flatten48(s["noisy"]))
                    np.save(os.path.join(args.outdir, f"{tag}_all_predictions.npy"), flatten48(M_noisy))

            row = dict(variant=variant, regime=regime,
                       noise_ratio=float(np.mean(nr)),
                       recovery=(float(np.mean(rec)) if rec else None),
                       recovery_lin=(float(np.mean(rec_lin)) if rec_lin else None),
                       mpjpe_in=float(np.mean(mi)), mpjpe_out=float(np.mean(mo)))
            summary.append(row)
            if row["recovery"] is not None:
                rec_str = f"net {row['recovery']*100:5.1f}% vs lin {row['recovery_lin']*100:5.1f}%"
            else:
                rec_str = "recovery n/a (realistic)"
            print(f"{variant:<10} {regime:<10} | {rec_str} | "
                  f"noise-ratio {row['noise_ratio']:.3f} | "
                  f"MPJPE {row['mpjpe_in']:.1f} -> {row['mpjpe_out']:.1f} mm")

    # --- headline interpretation ---
    print("\n=== reading the table ===")
    print("recovery -> 100% : the model removes the known offset (pure variant)")
    print("noise-ratio ~1   : honest harmonization;  <1 : masking real tracking error")
    print("nonlinear regime : morphing should beat a linear baseline here (run --regimes nonlinear)")
    print(f"\narrays for statistics/morphing_statistics.py written to {args.outdir}")


if __name__ == "__main__":
    main()
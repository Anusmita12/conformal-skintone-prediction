"""
Step 4 - Tone-blind conformal prediction under dataset shift.

Setting
  Malignancy model trained on a SOURCE dataset (PAD-UFES-20), deployed on a
  TARGET dataset (DDI). A small labelled target set is available for
  calibration, but at TEST time the patient's skin tone is NOT known.

Question
  Can we keep "at least 1 - alpha of cancers caught in EVERY skin tone"
  without observing skin tone at test time, and what does it cost?

Methods compared (all calibrated on the same target calibration set unless noted)
  Source-calibrated LAC   status quo: calibrate on source, ignore shift and tone
  Target marginal LAC     recalibrate on target, ignore tone
  Oracle tone x label     true tone known at test time (upper reference, not deployable)
  Plug-in tone x label    predicted tone used as if true            (no guarantee)
  Soft threshold          tone-probability-weighted threshold       (no guarantee)
  Worst-case union        every tone's threshold allowed            (valid, conservative)
  Tone-set union (ours)   conformal tone set at beta + label sets at alpha - beta (valid)
  Tone-adaptive (ours)    score rescaled by expected tone difficulty, worst-case calibrated (valid)

Skin-tone estimator (needed by plug-in / soft / ours)
  --tone_train ddi_split : train on a slice of the target pool (default; no extra data)
  --tone_train fitz17k   : train on Fitzpatrick17k (needs extract_extra.py features)
  --tone_features embed | ita | both

Usage
  python run_tone_blind.py --backbone dinov2_vits14
  python run_tone_blind.py --backbone dinov2_vits14 --tone_train fitz17k --tone_features both
  python run_tone_blind.py --backbone dinov2_vits14 --target fitz17k_ --tone_train ddi_split

Outputs: results/tone_blind/<tag>/alpha_<alpha>/
  trials.csv, summary.csv, beta_sweep.csv, reweighted.csv, below_target.csv, tone_model.csv
  main.png, beta_sweep.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import conformal as cp
import tone_blind as tb

TONE_TO_IDX = {12: 0, 34: 1, 56: 2}
TONE_NAMES = ["FST I-II", "FST III-IV", "FST V-VI"]
GROUPS = ["All"] + TONE_NAMES
ITA_COLS = ["ita_median", "ita_q25", "ita_q75", "L_median", "b_median"]

METHOD_ORDER = ["Source-calibrated LAC", "Target marginal LAC", "Oracle tone x label",
                "Plug-in tone x label", "Soft threshold", "Worst-case union",
                "Tone-set union (ours)", "Tone-adaptive (ours)"]
DEPLOYABLE_VALID = {"Target marginal LAC", "Worst-case union", "Tone-set union (ours)",
                    "Tone-adaptive (ours)"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load(features_dir, prefix, backbone, ita):
    X = np.load(features_dir / f"features_{prefix}{backbone}.npy")
    meta = pd.read_csv(features_dir / f"meta_{prefix}{backbone}.csv")
    if ita:
        f = features_dir / f"ita_{prefix}.csv"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing - run: python extract_extra.py --ita ...")
        ita_df = pd.read_csv(f).drop_duplicates("file")
        meta = meta.merge(ita_df, on="file", how="left")
        if meta[ITA_COLS].isna().any().any():
            raise ValueError(f"Some images in meta_{prefix}{backbone}.csv have no ITA row.")
    return X, meta


def tone_inputs(X, meta, mode):
    if mode == "embed":
        return X
    ita = meta[ITA_COLS].to_numpy()
    return ita if mode == "ita" else np.hstack([X, ita])


def fit_lr(X, y, C, seed):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000, class_weight="balanced",
                                            random_state=seed)).fit(X, y)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def group_rows(trial, method, sets, y, tone, w_dx=None, **extra):
    cov = cp.coverage(sets, y)
    size = cp.set_size(sets)
    single = tb.singleton_rate(sets)
    rows = []
    for g, name in [(None, "All")] + list(enumerate(TONE_NAMES)):
        m = np.ones(len(y), bool) if g is None else tone == g
        mal, ben = m & (y == 1), m & (y == 0)
        row = dict(trial=trial, method=method, group=name, n=int(m.sum()),
                   coverage=cov[m].mean(), set_size=size[m].mean(),
                   singleton_rate=single[m].mean(),
                   # cancers caught = malignant label inside the set
                   coverage_malignant=cov[mal].mean() if mal.any() else np.nan,
                   coverage_benign=cov[ben].mean() if ben.any() else np.nan,
                   **extra)
        if w_dx is not None:
            wm = w_dx * m
            row["coverage_dx_reweighted"] = (wm * cov).sum() / wm.sum() if wm.sum() > 0 else np.nan
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------
def run_trial(D, seed, args, tone_model_ext):
    rows, sweep, tone_rows, below = [], [], [], []
    a, beta = args.alpha, args.beta

    # ---- source model (patient-level split) ----
    gss = GroupShuffleSplit(n_splits=1, test_size=args.source_cal_frac, random_state=seed)
    s_tr, s_ca = next(gss.split(D["Xs"], D["ys"], groups=D["pid_s"]))
    clf = fit_lr(D["Xs"][s_tr], D["ys"][s_tr], args.C, seed)

    # ---- target split: pool / test ----
    yt, gt = D["yt"], D["gt"]
    pool, test = train_test_split(np.arange(len(yt)), test_size=args.target_test_frac,
                                  stratify=gt * 2 + yt, random_state=seed)

    # ---- tone estimator ----
    # part_A = labelled target data NOT used for calibration, for the tone-adaptive score
    if tone_model_ext is not None:
        tone_clf = tone_model_ext
        part_A, cal = train_test_split(pool, test_size=1 - args.adaptive_A_frac,
                                       stratify=gt[pool] * 2 + yt[pool], random_state=seed)
        cal_union = pool          # union-bound method can use the whole pool
    else:
        tone_tr, cal = train_test_split(pool, test_size=1 - args.tone_train_frac,
                                        stratify=gt[pool] * 2 + yt[pool], random_state=seed)
        tone_clf = fit_lr(D["Tt"][tone_tr], gt[tone_tr], args.C_tone, seed)
        part_A, cal_union = tone_tr, cal   # reuse tone-training slice as part A

    # every method except tone-adaptive calibrates on cal_union (largest valid set);
    # tone-adaptive calibrates on cal, with part_A fitting its score
    cal_adapt, cal = cal, cal_union
    pL_cal, pL_te = clf.predict_proba(D["Xt"][cal]), clf.predict_proba(D["Xt"][test])
    pT_cal, pT_te = tone_clf.predict_proba(D["Tt"][cal]), tone_clf.predict_proba(D["Tt"][test])
    y_cal, g_cal, y_te, g_te = yt[cal], gt[cal], yt[test], gt[test]
    w_te = D["w_dx"][test]

    tone_acc = (pT_te.argmax(1) == g_te)
    tone_rows.append(dict(trial=seed, tone_accuracy=tone_acc.mean(),
                          **{f"tone_acc_{n}": tone_acc[g_te == i].mean()
                             for i, n in enumerate(TONE_NAMES)},
                          malignancy_auc_target=roc_auc_score(y_te, pL_te[:, 1]),
                          n_cal=len(cal), n_test=len(test)))

    R = lambda method, sets, **kw: group_rows(seed, method, sets, y_te, g_te, w_te, **kw)

    # ---- status quo: source calibration ----
    p_src = clf.predict_proba(D["Xs"][s_ca])
    q = cp.conformal_quantile(cp.lac_scores(p_src, D["ys"][s_ca]), a)
    rows += R("Source-calibrated LAC", cp.lac_sets(pL_te, q))

    # ---- target marginal ----
    q = cp.conformal_quantile(cp.lac_scores(pL_cal, y_cal), a)
    rows += R("Target marginal LAC", cp.lac_sets(pL_te, q))

    # ---- tone-aware methods ----
    Q = tb.calibrate(pL_cal, y_cal, pT_cal, g_cal, a, beta, args.cells)
    rows += R("Oracle tone x label", tb.oracle_sets(pL_te, g_te, Q["qL_full"]))
    rows += R("Plug-in tone x label", tb.plugin_sets(pL_te, pT_te, Q["qL_full"]))
    rows += R("Soft threshold", tb.soft_threshold_sets(pL_te, pT_te, Q["qL_full"]))
    rows += R("Worst-case union", tb.worst_case_sets(pL_te, Q["qL_full"]))
    sets, tone_ok = tb.tone_set_union_sets(pL_te, pT_te, Q["qL_split"], Q["qT"],
                                           return_tone_sets=True)
    rows += R("Tone-set union (ours)", sets)
    tau0 = tb.tone_adaptive_fit(clf.predict_proba(D["Xt"][part_A]), yt[part_A], gt[part_A],
                                len(TONE_NAMES), a, args.cells)
    lam = tb.tone_adaptive_calibrate(clf.predict_proba(D["Xt"][cal_adapt]), yt[cal_adapt],
                                     tone_clf.predict_proba(D["Tt"][cal_adapt]), gt[cal_adapt],
                                     tau0, a, args.cells)
    rows += R("Tone-adaptive (ours)", tb.tone_adaptive_sets(pL_te, pT_te, tau0, lam))
    tone_rows[-1]["mean_tone_set_size_for_malignant"] = tone_ok[:, :, 1].sum(1).mean()
    tone_rows[-1]["n_infinite_label_cells"] = int(np.isinf(Q["qL_split"]).sum())

    # ---- beta sweep for the trade-off curve ----
    for b in args.beta_list:
        if b >= a:
            continue
        Qb = tb.calibrate(pL_cal, y_cal, pT_cal, g_cal, a, b, args.cells)
        s = tb.tone_set_union_sets(pL_te, pT_te, Qb["qL_split"], Qb["qT"])
        sweep += group_rows(seed, "Tone-set union (ours)", s, y_te, g_te, beta=b)

    # ---- expected 'below target' rate for an exactly valid method ----
    # reference for oracle tone x label coverage on MALIGNANT cases in each tone
    for i, n in enumerate(TONE_NAMES):
        n_c = int(((g_cal == i) & (y_cal == 1)).sum())
        n_t = int(((g_te == i) & (y_te == 1)).sum())
        below.append(dict(trial=seed, group=n, n_cal_malignant=n_c, n_test_malignant=n_t,
                          expected_below=tb.expected_below_target(n_c, n_t, a)))
    return rows, sweep, tone_rows, below


# ---------------------------------------------------------------------------
# Summaries and plots
# ---------------------------------------------------------------------------
def summarise(df, keys, alpha):
    cols = [c for c in ["coverage", "coverage_malignant", "coverage_benign", "set_size",
                        "singleton_rate", "coverage_dx_reweighted"] if c in df]
    g = df.groupby(keys, sort=False)
    out = g[cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["pct_trials_malignant_below_target"] = g["coverage_malignant"].apply(
        lambda c: (c < 1 - alpha).mean() * 100)
    return out.reset_index()


def fmt(m, s):
    return m.map("{:.3f}".format) + " ± " + s.map("{:.3f}".format)


def plot_main(summary, alpha, title, path):
    methods = [m for m in METHOD_ORDER if m in set(summary.method)]
    x, w = np.arange(len(TONE_NAMES)), 0.8 / len(methods)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cols = [("coverage_malignant", "Cancers caught (coverage on malignant)"),
            ("set_size", "Average set size"),
            ("singleton_rate", "Singleton rate (committed predictions)")]
    for ax, (c, lab) in zip(axes, cols):
        for i, m in enumerate(methods):
            d = summary[summary.method == m].set_index("group").reindex(TONE_NAMES)
            hatch = "//" if m not in DEPLOYABLE_VALID and m != "Oracle tone x label" else None
            ax.bar(x + i * w, d[f"{c}_mean"], w, yerr=d[f"{c}_std"], capsize=2,
                   label=m, hatch=hatch, edgecolor="white" if hatch else None)
        if c == "coverage_malignant":
            ax.axhline(1 - alpha, color="k", ls="--", lw=1)
            ax.set_ylim(0, 1.05)
        ax.set_xticks(x + w * (len(methods) - 1) / 2, TONE_NAMES)
        ax.set_title(lab)
        ax.grid(axis="y", alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(title + "   (hatched = no coverage guarantee)", fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep(sw, summary, alpha, path):
    """Worst-tone cancer coverage vs set size, ours across beta + reference points."""
    fig, ax = plt.subplots(figsize=(7, 5))
    tone_only = sw[sw.group.isin(TONE_NAMES)]
    worst = tone_only.groupby("beta").coverage_malignant_mean.min()
    size = sw[sw.group == "All"].set_index("beta").set_size_mean
    ax.plot(size.loc[worst.index], worst, "o-", color="C6", label="Tone-set union, varying beta")
    for b in worst.index:
        ax.annotate(f"β={b:g}", (size.loc[b], worst.loc[b]), fontsize=7,
                    xytext=(4, -10), textcoords="offset points")
    for m, mk in [("Oracle tone x label", "*"), ("Plug-in tone x label", "x"),
                  ("Worst-case union", "s"), ("Target marginal LAC", "^"),
                  ("Soft threshold", "d")]:
        d = summary[summary.method == m]
        if d.empty:
            continue
        ax.scatter(d[d.group == "All"].set_size_mean,
                   d[d.group.isin(TONE_NAMES)].coverage_malignant_mean.min(),
                   marker=mk, s=80, label=m)
    ax.axhline(1 - alpha, color="k", ls="--", lw=1)
    ax.set_xlabel("Average set size (all test images)")
    ax.set_ylabel("Cancers caught in the WORST skin-tone group")
    ax.set_title("Price of not knowing skin tone")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--features_dir", type=Path, default=Path("features"))
    ap.add_argument("--out_dir", type=Path, default=Path("results/tone_blind"))
    ap.add_argument("--source", default="pad_ufes_", help="source feature prefix")
    ap.add_argument("--target", default="", help="target prefix: '' = DDI, 'fitz17k_' = Fitzpatrick17k")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=None, help="tone budget (default alpha/2)")
    ap.add_argument("--beta_list", type=float, nargs="+",
                    default=[0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.09])
    ap.add_argument("--cells", default="tone_label", choices=["tone_label", "tone"],
                    help="guarantee per (tone, label) cell or per tone only")
    ap.add_argument("--tone_train", default="ddi_split", choices=["ddi_split", "fitz17k"])
    ap.add_argument("--tone_features", default="embed", choices=["embed", "ita", "both"])
    ap.add_argument("--tone_train_frac", type=float, default=0.3,
                    help="ddi_split: share of the target pool used to train the tone model")
    ap.add_argument("--adaptive_A_frac", type=float, default=0.4,
                    help="fitz17k tone model: share of the target pool used to fit the "
                         "tone-adaptive score (rest calibrates it)")
    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--source_cal_frac", type=float, default=0.2)
    ap.add_argument("--target_test_frac", type=float, default=0.4)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--C_tone", type=float, default=0.1)
    ap.add_argument("--min_dx_per_tone", type=int, default=2)
    args = ap.parse_args()
    args.beta = args.alpha / 2 if args.beta is None else args.beta
    assert 0 < args.beta < args.alpha, "need 0 < beta < alpha"

    use_ita = args.tone_features in ("ita", "both")
    Xs, ms = load(args.features_dir, args.source, args.backbone, ita=False)
    Xt, mt = load(args.features_dir, args.target, args.backbone, ita=use_ita)
    keep = mt.skin_tone.isin(TONE_TO_IDX).to_numpy()            # drop unknown tone
    Xt, mt = Xt[keep], mt[keep].reset_index(drop=True)

    D = dict(Xs=Xs, ys=ms.malignant.to_numpy().astype(int), pid_s=ms.patient_id.to_numpy(),
             Xt=Xt, yt=mt.malignant.to_numpy().astype(int),
             gt=mt.skin_tone.map(TONE_TO_IDX).to_numpy(),
             Tt=tone_inputs(Xt, mt, args.tone_features))
    w_dx, shared = tb.diagnosis_weights(D["gt"], mt.disease.fillna("").to_numpy(),
                                        args.min_dx_per_tone)
    D["w_dx"] = w_dx

    tone_model_ext = None
    if args.tone_train == "fitz17k":
        if args.target.startswith("fitz17k"):
            raise SystemExit("Target is Fitzpatrick17k: use --tone_train ddi_split.")
        fz = "fitz17k_all_" if (args.features_dir /
                                 f"features_fitz17k_all_{args.backbone}.npy").exists() else "fitz17k_"
        Xf, mf = load(args.features_dir, fz, args.backbone, ita=use_ita)
        kf = mf.skin_tone.isin(TONE_TO_IDX).to_numpy()
        Tf = tone_inputs(Xf[kf], mf[kf].reset_index(drop=True), args.tone_features)
        tone_model_ext = fit_lr(Tf, mf.skin_tone[kf].map(TONE_TO_IDX).to_numpy(), args.C_tone, 0)
        print(f"Tone estimator trained on Fitzpatrick17k ({fz}): {kf.sum()} images")

    print(f"Source {args.source!r}: {len(D['ys'])} images | Target {args.target or 'ddi'!r}: "
          f"{len(D['yt'])} images, {D['yt'].mean():.0%} malignant")
    print(f"alpha={args.alpha}  beta={args.beta}  cells={args.cells}  "
          f"tone model: {args.tone_train}/{args.tone_features}")
    print(f"Diagnosis reweighting uses {len(shared)} diagnoses shared by all tones "
          f"({(w_dx > 0).sum()} of {len(w_dx)} target images)\n")

    rows, sweep, tone_rows, below = [], [], [], []
    for seed in range(args.n_trials):
        r, s, t, b = run_trial(D, seed, args, tone_model_ext)
        rows += r; sweep += s; tone_rows += t; below += b
    trials, sweep, tone_df, below = map(pd.DataFrame, (rows, sweep, tone_rows, below))

    summary = summarise(trials, ["method", "group"], args.alpha)
    summary["method"] = pd.Categorical(summary.method, METHOD_ORDER, ordered=True)
    summary = summary.sort_values(["method", "group"]).reset_index(drop=True)
    summary["method"] = summary.method.astype(str)
    sw = summarise(sweep, ["beta", "group"], args.alpha)

    # observed vs expected 'below target' for the oracle (an exactly valid method)
    orc = trials[(trials.method == "Oracle tone x label") & trials.group.isin(TONE_NAMES)]
    below_tab = (below.groupby("group").agg(n_cal_malignant=("n_cal_malignant", "mean"),
                                            n_test_malignant=("n_test_malignant", "mean"),
                                            expected_pct=("expected_below", "mean"))
                 .assign(expected_pct=lambda d: d.expected_pct * 100))
    below_tab["observed_pct_oracle"] = orc.groupby("group").coverage_malignant.apply(
        lambda c: (c < 1 - args.alpha).mean() * 100)

    tag = f"{args.backbone}_{args.source.strip('_')}_to_{args.target.strip('_') or 'ddi'}_" \
          f"tone-{args.tone_train}-{args.tone_features}_{args.cells}"
    out = args.out_dir / tag / f"alpha_{args.alpha}"
    out.mkdir(parents=True, exist_ok=True)
    trials.to_csv(out / "trials.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    sw.to_csv(out / "beta_sweep.csv", index=False)
    below_tab.to_csv(out / "below_target.csv")
    tone_df.describe().T.to_csv(out / "tone_model.csv")
    if (w_dx > 0).any():
        summary.pivot(index="method", columns="group", values="coverage_dx_reweighted_mean") \
            .reindex(METHOD_ORDER)[TONE_NAMES].to_csv(out / "reweighted.csv")

    # ---- console report ----
    pd.set_option("display.width", 200)
    print("=== Tone estimator & base model (mean over trials) ===")
    print(tone_df.drop(columns="trial").mean().round(3).to_string())
    for title, col in [("Cancers caught (coverage on malignant)", "coverage_malignant"),
                       ("Overall coverage", "coverage"),
                       ("Average set size", "set_size"),
                       ("Singleton rate", "singleton_rate")]:
        t = summary.assign(v=fmt(summary[f"{col}_mean"], summary[f"{col}_std"]))
        print(f"\n=== {title} (target {1 - args.alpha:.0%}) ===")
        print(t.pivot(index="method", columns="group", values="v").reindex(METHOD_ORDER)[GROUPS]
              .to_string())
    if "coverage_dx_reweighted_mean" in summary:
        print("\n=== Overall coverage, diagnosis-mix reweighted (same disease mix in every tone) ===")
        print(summary.pivot(index="method", columns="group", values="coverage_dx_reweighted_mean")
              .reindex(METHOD_ORDER)[TONE_NAMES].round(3).to_string())
    print("\n=== % trials with cancer coverage below target: observed (oracle) vs expected "
          "for an exactly valid method ===")
    print(below_tab.round(1).to_string())
    print("\n=== Beta sweep (tone-set union): cancers caught / set size ===")
    piv = sw.assign(v=sw.coverage_malignant_mean.round(3).astype(str) + " / "
                    + sw.set_size_mean.round(2).astype(str))
    print(piv.pivot(index="beta", columns="group", values="v")[GROUPS].to_string())

    title = (f"{args.source.strip('_')} -> {args.target.strip('_') or 'DDI'}, "
             f"alpha={args.alpha}, beta={args.beta}")
    plot_main(summary, args.alpha, title, out / "main.png")
    plot_sweep(sw, summary, args.alpha, out / "beta_sweep.png")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()

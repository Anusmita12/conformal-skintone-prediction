"""
Step 3 - Conformal prediction under a skin-tone distribution shift.

Setting (mimics real deployment):
  * A malignancy classifier is trained on PAD-UFES-20 (source: Brazil,
    smartphone photos, mostly FST I-III).
  * It is deployed on DDI (target: balanced FST I-VI, biopsy-proven).

Questions
  Q1  If conformal prediction is calibrated on SOURCE data (the usual
      practice), does the coverage guarantee still hold on DDI, and for which
      skin tones does it break?
  Q2  Can it be repaired by recalibrating on a small set of TARGET (DDI)
      images - pooled (marginal) vs per skin tone (Mondrian)?
  Q3  How many labelled target images per skin tone are needed, and what is
      the cost in prediction-set size?

Each trial re-draws a patient-level PAD-UFES train/calibration split and a
DDI calibration-pool/test split, retrains the classifier, and evaluates.

Usage (after extracting features for both datasets with the same backbone):
    python run_shift.py --backbone dinov2_vits14
    python run_shift.py --backbone dinov2_vits14 --alpha 0.1 --n_trials 100

Outputs in results/shift/<backbone>/alpha_<alpha>/:
    trials.csv, summary.csv, fewshot.csv,
    shift_coverage.png  (Q1 + Q2),  fewshot.png  (Q3)
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

TONE_TO_IDX = {12: 0, 34: 1, 56: 2}
TONE_NAMES = ["FST I-II", "FST III-IV", "FST V-VI"]
GROUPS = ["All"] + TONE_NAMES


def load(features_dir, prefix, backbone):
    X = np.load(features_dir / f"features_{prefix}{backbone}.npy")
    meta = pd.read_csv(features_dir / f"meta_{prefix}{backbone}.csv")
    return X, meta


def fit_classifier(X, y, C, seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=5000, class_weight="balanced",
                           random_state=seed)).fit(X, y)


def group_rows(trial, method, sets, y, tone, **extra):
    """Coverage / set size overall and per skin-tone group."""
    cov, size = cp.coverage(sets, y), cp.set_size(sets)
    rows = []
    for g, name in [(None, "All")] + list(enumerate(TONE_NAMES)):
        m = np.ones(len(y), bool) if g is None else tone == g
        rows.append(dict(trial=trial, method=method, group=name, n=int(m.sum()),
                         coverage=cov[m].mean(), set_size=size[m].mean(),
                         # coverage on malignant cases only = "missed cancer" view
                         coverage_malignant=cov[m & (y == 1)].mean()
                         if (m & (y == 1)).any() else np.nan,
                         **extra))
    return rows


def sample_per_tone(idx, tone, k, rng):
    """k random indices from each skin-tone group (or all if fewer exist)."""
    out = []
    for g in range(3):
        pool = idx[tone[idx] == g]
        out.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
    return np.concatenate(out)


def run_trial(Xs, ys, pid_s, Xt, yt, tone_t, seed, args):
    rng = np.random.default_rng(seed)
    rows, few, base = [], [], []

    # ---- source: patient-level split (no patient in both train and cal) ----
    gss = GroupShuffleSplit(n_splits=1, test_size=args.source_cal_frac, random_state=seed)
    s_tr, s_ca = next(gss.split(Xs, ys, groups=pid_s))
    clf = fit_classifier(Xs[s_tr], ys[s_tr], args.C, seed)

    # ---- target: DDI calibration pool / test split -------------------------
    t_pool, t_te = train_test_split(np.arange(len(yt)), test_size=args.target_test_frac,
                                    stratify=tone_t * 2 + yt, random_state=seed)

    p_src_cal = clf.predict_proba(Xs[s_ca])
    p_pool = clf.predict_proba(Xt[t_pool])
    p_te = clf.predict_proba(Xt[t_te])
    y_te, tone_te = yt[t_te], tone_t[t_te]

    # base-model discrimination: in-domain (PAD cal) vs shifted (DDI test)
    base.append(dict(trial=seed, group="PAD-UFES (in-domain)",
                     auc=roc_auc_score(ys[s_ca], p_src_cal[:, 1])))
    for g, name in [(None, "DDI All")] + [(i, f"DDI {n}") for i, n in enumerate(TONE_NAMES)]:
        m = np.ones(len(y_te), bool) if g is None else tone_te == g
        base.append(dict(trial=seed, group=name, auc=roc_auc_score(y_te[m], p_te[m, 1])))

    a = args.alpha
    # Q1: calibrate on SOURCE, test on target
    q = cp.conformal_quantile(cp.lac_scores(p_src_cal, ys[s_ca]), a)
    rows += group_rows(seed, "Source-calibrated LAC", cp.lac_sets(p_te, q), y_te, tone_te)
    q = cp.conformal_quantile(cp.aps_scores(p_src_cal, ys[s_ca], rng), a)
    rows += group_rows(seed, "Source-calibrated APS", cp.aps_sets(p_te, q, rng), y_te, tone_te)

    # Q2: recalibrate on the full TARGET calibration pool
    y_pool, tone_pool = yt[t_pool], tone_t[t_pool]
    q = cp.conformal_quantile(cp.lac_scores(p_pool, y_pool), a)
    rows += group_rows(seed, "Target-calibrated LAC", cp.lac_sets(p_te, q), y_te, tone_te)
    qs = cp.mondrian_lac_qhats(p_pool, y_pool, tone_pool, a)
    rows += group_rows(seed, "Target Mondrian LAC (by tone)",
                       cp.mondrian_lac_sets(p_te, tone_te, qs), y_te, tone_te)
    # tone x label: guarantees coverage of malignant cases within every tone
    qs = cp.group_label_lac_qhats(p_pool, y_pool, tone_pool, a)
    rows += group_rows(seed, "Target tone x label LAC",
                       cp.group_label_lac_sets(p_te, tone_te, qs), y_te, tone_te)

    # Q3: few-shot target calibration, k labelled images per skin tone
    pool_pos = np.arange(len(t_pool))
    for k in args.k_list:
        sel = sample_per_tone(pool_pos, tone_pool, k, rng)
        p_k, y_k, g_k = p_pool[sel], y_pool[sel], tone_pool[sel]
        q = cp.conformal_quantile(cp.lac_scores(p_k, y_k), a)
        for r in group_rows(seed, "Pooled (marginal)", cp.lac_sets(p_te, q), y_te, tone_te, k=k):
            few.append(r)
        qs = cp.mondrian_lac_qhats(p_k, y_k, g_k, a)
        for r in group_rows(seed, "Per-tone (Mondrian)",
                            cp.mondrian_lac_sets(p_te, tone_te, qs), y_te, tone_te, k=k):
            few.append(r)
    return rows, few, base


def summarise(df, keys, alpha):
    g = df.groupby(keys, sort=False)
    out = g[["coverage", "set_size", "coverage_malignant"]].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    # share of trials in which this group fell below the target coverage
    out["pct_trials_below_target"] = g["coverage"].apply(lambda c: (c < 1 - alpha).mean() * 100)
    return out.reset_index()


def fmt(m, s):
    return m.map("{:.3f}".format) + " ± " + s.map("{:.3f}".format)


def plot_shift(summary, alpha, path):
    methods = list(dict.fromkeys(summary.method))
    x, w = np.arange(len(GROUPS)), 0.8 / len(methods)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for i, m in enumerate(methods):
        d = summary[summary.method == m].set_index("group").reindex(GROUPS)
        axes[0].bar(x + i * w, d.coverage_mean, w, yerr=d.coverage_std, capsize=3, label=m)
        axes[1].bar(x + i * w, d.set_size_mean, w, yerr=d.set_size_std, capsize=3, label=m)
    axes[0].axhline(1 - alpha, color="k", ls="--", lw=1, label=f"Target {1 - alpha:.0%}")
    axes[0].set_ylim(max(0, summary.coverage_mean.min() - 0.15), 1.02)
    axes[0].set_ylabel("Empirical coverage on DDI test")
    axes[1].set_ylabel("Average set size")
    for ax, t in zip(axes, ["Coverage by skin tone", "Set size by skin tone"]):
        ax.set_xticks(x + w * (len(methods) - 1) / 2, GROUPS)
        ax.set_title(t)
        ax.grid(axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=5)
    fig.suptitle(f"Train on PAD-UFES-20 -> deploy on DDI  (malignancy, alpha = {alpha})",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fewshot(fs, alpha, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = dict(zip(TONE_NAMES, ["#e6a96b", "#a0673a", "#4a2c17"]))
    for ax, m in zip(axes[:2], ["Pooled (marginal)", "Per-tone (Mondrian)"]):
        for gname in TONE_NAMES:
            d = fs[(fs.method == m) & (fs.group == gname)].sort_values("k")
            ax.errorbar(d.k, d.coverage_mean, yerr=d.coverage_std, marker="o",
                        capsize=3, color=colors[gname], label=gname)
        ax.axhline(1 - alpha, color="k", ls="--", lw=1)
        ax.set_title(f"{m} calibration")
        ax.set_xlabel("Labelled DDI images per skin tone (k)")
        ax.set_ylabel("Coverage on DDI test")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(fs.k.unique()), [str(k) for k in sorted(fs.k.unique())])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for m, ls in [("Pooled (marginal)", "--"), ("Per-tone (Mondrian)", "-")]:
        for gname in TONE_NAMES:
            d = fs[(fs.method == m) & (fs.group == gname)].sort_values("k")
            axes[2].plot(d.k, d.set_size_mean, ls=ls, marker="o", color=colors[gname],
                         label=f"{gname} - {m.split()[0]}")
    axes[2].set_title("Set size (dashed = pooled, solid = per-tone)")
    axes[2].set_xlabel("Labelled DDI images per skin tone (k)")
    axes[2].set_ylabel("Average set size")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(sorted(fs.k.unique()), [str(k) for k in sorted(fs.k.unique())])
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=7)
    fig.suptitle(f"How much target data restores the guarantee?  (alpha = {alpha})",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--features_dir", type=Path, default=Path("features"))
    ap.add_argument("--out_dir", type=Path, default=Path("results/shift"))
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--source_cal_frac", type=float, default=0.2)
    ap.add_argument("--target_test_frac", type=float, default=0.5)
    ap.add_argument("--k_list", type=int, nargs="+", default=[10, 20, 40, 80])
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--source", default="pad_ufes",
                    help="source feature prefix, e.g. pad_ufes or pad_ufes_ackexcl")
    args = ap.parse_args()

    Xs, ms = load(args.features_dir, f"{args.source}_", args.backbone)
    Xt, mt = load(args.features_dir, "", args.backbone)
    ys, pid_s = ms.malignant.to_numpy().astype(int), ms.patient_id.to_numpy()
    yt = mt.malignant.to_numpy().astype(int)
    tone_t = mt.skin_tone.map(TONE_TO_IDX).to_numpy()

    src_tones = ms.skin_tone.map({12: "FST I-II", 34: "FST III-IV", 56: "FST V-VI",
                                  0: "unknown"}).value_counts()
    print(f"Source PAD-UFES: {len(ys)} images, {ys.mean():.0%} malignant, "
          f"{len(np.unique(pid_s))} patients | skin tones: {src_tones.to_dict()}")
    print(f"Target DDI:      {len(yt)} images, {yt.mean():.0%} malignant")
    print(f"alpha={args.alpha} | {args.n_trials} trials | k per tone = {args.k_list}\n")

    rows, few, base = [], [], []
    for seed in range(args.n_trials):
        r, f, b = run_trial(Xs, ys, pid_s, Xt, yt, tone_t, seed, args)
        rows += r; few += f; base += b
    trials, fewshot, base = pd.DataFrame(rows), pd.DataFrame(few), pd.DataFrame(base)

    summary = summarise(trials, ["method", "group"], args.alpha)
    fs_summary = summarise(fewshot, ["method", "k", "group"], args.alpha)

    sub = args.backbone if args.source == "pad_ufes" else f"{args.backbone}_{args.source}"
    out = args.out_dir / sub / f"alpha_{args.alpha}"
    out.mkdir(parents=True, exist_ok=True)
    trials.to_csv(out / "trials.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    fs_summary.to_csv(out / "fewshot.csv", index=False)
    base.groupby("group", sort=False).auc.agg(["mean", "std"]).to_csv(out / "auc.csv")

    pd.set_option("display.width", 160)
    print("=== Base model AUC (malignancy) ===")
    print(base.groupby("group", sort=False).auc.agg(["mean", "std"]).round(3).to_string())

    for title, col in [("Coverage", "coverage"), ("Set size", "set_size"),
                       ("Coverage on malignant cases only", "coverage_malignant")]:
        t = summary.assign(v=fmt(summary[f"{col}_mean"], summary[f"{col}_std"]))
        print(f"\n=== {title} on DDI test (target {1 - args.alpha:.0%}) ===")
        print(t.pivot(index="method", columns="group", values="v")[GROUPS].to_string())

    print(f"\n=== % of trials below {1 - args.alpha:.0%} coverage ===")
    print(summary.pivot(index="method", columns="group", values="pct_trials_below_target")
          [GROUPS].round(0).to_string())

    print("\n=== Few-shot target calibration: coverage (k images per tone) ===")
    t = fs_summary.assign(v=fmt(fs_summary.coverage_mean, fs_summary.coverage_std))
    print(t.pivot_table(index=["method", "k"], columns="group", values="v", aggfunc="first")
          [GROUPS].to_string())

    plot_shift(summary, args.alpha, out / "shift_coverage.png")
    plot_fewshot(fs_summary, args.alpha, out / "fewshot.png")
    print(f"\nSaved results and plots to {out}/")


if __name__ == "__main__":
    main()
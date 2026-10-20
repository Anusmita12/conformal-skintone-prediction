"""
Step 2 - Conformal prediction experiments on DDI embeddings.

Two tasks, both evaluated per Fitzpatrick skin-tone group
(FST I-II, III-IV, V-VI):

  Task A  skin_tone   predict the FST group (3 classes) from the image.
          Methods: LAC, APS (marginal) vs class-conditional LAC.

  Task B  malignancy  predict benign vs malignant.
          Methods: LAC, APS (marginal) vs Mondrian LAC calibrated
          separately per skin-tone group.

Research question: does *marginal* conformal coverage (e.g. 90% overall)
hide under-coverage for some skin tones, and what does group-conditional
calibration cost in set size?

Each run repeats a random stratified split (train / calibration / test)
--n_trials times and reports mean +/- std, because DDI is small.

Usage:
    python run_experiment.py --backbone resnet50
    python run_experiment.py --backbone dinov2_vits14 --alpha 0.1 --n_trials 100

Outputs in results/<backbone>/:
    trials.csv, summary.csv, coverage_skin_tone.png, coverage_malignancy.png
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import conformal as cp

TONE_TO_IDX = {12: 0, 34: 1, 56: 2}
TONE_NAMES = ["FST I-II", "FST III-IV", "FST V-VI"]


def split_indices(strata, seed, cal_frac, test_frac):
    idx = np.arange(len(strata))
    train, rest = train_test_split(idx, test_size=cal_frac + test_frac,
                                   stratify=strata, random_state=seed)
    cal, test = train_test_split(rest, test_size=test_frac / (cal_frac + test_frac),
                                 stratify=strata[rest], random_state=seed)
    return train, cal, test


def fit_classifier(X, y, C, seed):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=5000, class_weight="balanced",
                           random_state=seed))
    return clf.fit(X, y)


def per_group_rows(trial, task, method, sets, y, tone, extra=None):
    """One row for 'All' plus one per skin-tone group."""
    cov, size = cp.coverage(sets, y), cp.set_size(sets)
    rows = []
    for g, name in [(None, "All")] + list(enumerate(TONE_NAMES)):
        m = np.ones(len(y), bool) if g is None else tone == g
        if m.sum() == 0:
            continue
        rows.append(dict(trial=trial, task=task, method=method, group=name,
                         n=int(m.sum()), coverage=cov[m].mean(),
                         set_size=size[m].mean(), **(extra or {})))
    return rows


def base_model_rows(trial, task, probs, y, tone):
    pred = probs.argmax(1)
    rows = []
    for g, name in [(None, "All")] + list(enumerate(TONE_NAMES)):
        m = np.ones(len(y), bool) if g is None else tone == g
        acc = (pred[m] == y[m]).mean()
        auc = np.nan
        if task == "malignancy" and len(np.unique(y[m])) == 2:
            auc = roc_auc_score(y[m], probs[m, 1])
        rows.append(dict(trial=trial, task=task, method="Base model", group=name,
                         n=int(m.sum()), accuracy=acc, auc=auc))
    return rows


def run_trial(X, tone, malignant, seed, args):
    rng = np.random.default_rng(seed)
    strata = tone * 2 + malignant                     # keep tone x label balance
    tr, ca, te = split_indices(strata, seed, args.cal_frac, args.test_frac)
    rows = []

    # ---------------- Task A: skin-tone prediction ------------------------
    y = tone
    clf = fit_classifier(X[tr], y[tr], args.C, seed)
    p_cal, p_test = clf.predict_proba(X[ca]), clf.predict_proba(X[te])
    rows += base_model_rows(seed, "skin_tone", p_test, y[te], tone[te])

    q = cp.conformal_quantile(cp.lac_scores(p_cal, y[ca]), args.alpha)
    rows += per_group_rows(seed, "skin_tone", "LAC", cp.lac_sets(p_test, q), y[te], tone[te])

    q = cp.conformal_quantile(cp.aps_scores(p_cal, y[ca], rng), args.alpha)
    rows += per_group_rows(seed, "skin_tone", "APS", cp.aps_sets(p_test, q, rng), y[te], tone[te])

    qs = cp.mondrian_lac_qhats(p_cal, y[ca], y[ca], args.alpha)
    rows += per_group_rows(seed, "skin_tone", "Class-cond. LAC",
                           cp.class_conditional_lac_sets(p_test, qs), y[te], tone[te])

    # ---------------- Task B: malignancy, groups = skin tone --------------
    y = malignant
    clf = fit_classifier(X[tr], y[tr], args.C, seed)
    p_cal, p_test = clf.predict_proba(X[ca]), clf.predict_proba(X[te])
    rows += base_model_rows(seed, "malignancy", p_test, y[te], tone[te])

    q = cp.conformal_quantile(cp.lac_scores(p_cal, y[ca]), args.alpha)
    rows += per_group_rows(seed, "malignancy", "LAC", cp.lac_sets(p_test, q), y[te], tone[te])

    q = cp.conformal_quantile(cp.aps_scores(p_cal, y[ca], rng), args.alpha)
    rows += per_group_rows(seed, "malignancy", "APS", cp.aps_sets(p_test, q, rng), y[te], tone[te])

    qs = cp.mondrian_lac_qhats(p_cal, y[ca], tone[ca], args.alpha)
    rows += per_group_rows(seed, "malignancy", "Mondrian LAC (by tone)",
                           cp.mondrian_lac_sets(p_test, tone[te], qs), y[te], tone[te])
    return rows


def plot_task(summary, task, alpha, out_path):
    s = summary[(summary.task == task) & (summary.method != "Base model")]
    methods = list(dict.fromkeys(s.method))
    groups = ["All"] + TONE_NAMES
    x = np.arange(len(groups))
    w = 0.8 / len(methods)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, m in enumerate(methods):
        d = s[s.method == m].set_index("group").reindex(groups)
        axes[0].bar(x + i * w, d.coverage_mean, w, yerr=d.coverage_std, capsize=3, label=m)
        axes[1].bar(x + i * w, d.set_size_mean, w, yerr=d.set_size_std, capsize=3, label=m)
    axes[0].axhline(1 - alpha, color="k", ls="--", lw=1, label=f"Target {1 - alpha:.0%}")
    axes[0].set_ylim(max(0, 1 - alpha - 0.35), 1.02)
    axes[0].set_ylabel("Empirical coverage")
    axes[1].set_ylabel("Average set size")
    for ax, t in zip(axes, ["Coverage by skin-tone group", "Set size by skin-tone group"]):
        ax.set_xticks(x + w * (len(methods) - 1) / 2, groups)
        ax.set_title(t)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    fig.suptitle(f"DDI - task: {task}  (alpha = {alpha})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--features_dir", type=Path, default=Path("features"))
    ap.add_argument("--out_dir", type=Path, default=Path("results"))
    ap.add_argument("--alpha", type=float, default=0.1, help="miscoverage level")
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--cal_frac", type=float, default=0.25)
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--C", type=float, default=0.1, help="inverse L2 strength")
    args = ap.parse_args()

    X = np.load(args.features_dir / f"features_{args.backbone}.npy")
    meta = pd.read_csv(args.features_dir / f"meta_{args.backbone}.csv")
    tone = meta.skin_tone.map(TONE_TO_IDX).to_numpy()
    malignant = meta.malignant.to_numpy().astype(int)
    print(f"{len(X)} images, {X.shape[1]}-d features | alpha={args.alpha} | "
          f"{args.n_trials} random splits")

    rows = []
    for seed in range(args.n_trials):
        rows += run_trial(X, tone, malignant, seed, args)
    trials = pd.DataFrame(rows)

    metrics = [c for c in ["coverage", "set_size", "accuracy", "auc"] if c in trials]
    summary = (trials.groupby(["task", "method", "group"], sort=False)[metrics]
               .agg(["mean", "std"]))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()

    out = args.out_dir / args.backbone
    out.mkdir(parents=True, exist_ok=True)
    trials.to_csv(out / "trials.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)

    # readable console tables
    pd.set_option("display.width", 140)
    for task in ["skin_tone", "malignancy"]:
        s = summary[summary.task == task]
        print(f"\n=== Task: {task} ===")
        base = s[s.method == "Base model"]
        cols = ["accuracy_mean"] + (["auc_mean"] if task == "malignancy" else [])
        print(base.set_index("group")[cols].round(3).T.to_string())
        conf = s[s.method != "Base model"]
        for metric in ["coverage", "set_size"]:
            tbl = conf.assign(val=conf[f"{metric}_mean"].map("{:.3f}".format) + " ± "
                              + conf[f"{metric}_std"].map("{:.3f}".format))
            print(f"\n{metric}:")
            print(tbl.pivot(index="method", columns="group", values="val")
                  [["All"] + TONE_NAMES].to_string())
        plot_task(summary, task, args.alpha, out / f"coverage_{task}.png")

    print(f"\nSaved results and plots to {out}/")


if __name__ == "__main__":
    main()
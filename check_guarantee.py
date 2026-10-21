"""
Sanity check on synthetic data (no images needed, runs in seconds).

We build a world where we KNOW the truth:
  * 3 skin-tone groups; the malignancy model is noticeably worse on group 2
  * a skin-tone estimator whose accuracy we control with --tone_signal
Then check, over many random calibration/test draws, which methods keep
"at least 1 - alpha of cancers caught in every tone".

Expected outcome
  Oracle, Worst-case union, both 'ours'    -> >= 0.90 in every tone
  Plug-in, Soft threshold                  -> can fall below 0.90 for the hard tone
  Tone-set union sets shrink toward plug-in size as --tone_signal grows

    python check_guarantee.py
    python check_guarantee.py --tone_signal 4     # tone very predictable
    python check_guarantee.py --tone_signal 0.3   # tone barely predictable
"""

import argparse

import numpy as np
import pandas as pd

import tone_blind as tb

NAMES = ["tone 0", "tone 1", "tone 2 (hard)"]


def simulate(n, rng, tone_signal, label_signal=(2.0, 1.6, 0.6), base_rate=0.3):
    g = rng.integers(0, 3, n)
    y = (rng.random(n) < base_rate).astype(int)
    # malignancy model: weaker signal for the hard tone
    s = np.array(label_signal)[g]
    logit = s * (2 * y - 1) + rng.normal(0, 1, n)
    p1 = 1 / (1 + np.exp(-logit))
    p_label = np.stack([1 - p1, p1], 1)
    # tone model: noisy logits around the true tone
    z = rng.normal(0, 1, (n, 3))
    z[np.arange(n), g] += tone_signal
    p_tone = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return p_label, y, p_tone, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--tone_signal", type=float, default=2.0)
    ap.add_argument("--n_cal", type=int, default=600)
    ap.add_argument("--n_test", type=int, default=3000)
    ap.add_argument("--trials", type=int, default=300)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    rows = []
    for t in range(args.trials):
        pLc, yc, pTc, gc = simulate(args.n_cal, rng, args.tone_signal)
        pLt, yt, pTt, gt = simulate(args.n_test, rng, args.tone_signal)
        Q = tb.calibrate(pLc, yc, pTc, gc, args.alpha, args.beta)
        pLa, ya, _, ga = simulate(args.n_cal, rng, args.tone_signal)   # part A for adaptive
        tau0 = tb.tone_adaptive_fit(pLa, ya, ga, 3, args.alpha)
        lam = tb.tone_adaptive_calibrate(pLc, yc, pTc, gc, tau0, args.alpha)
        methods = {
            "Oracle tone x label": tb.oracle_sets(pLt, gt, Q["qL_full"]),
            "Plug-in tone x label": tb.plugin_sets(pLt, pTt, Q["qL_full"]),
            "Soft threshold": tb.soft_threshold_sets(pLt, pTt, Q["qL_full"]),
            "Worst-case union": tb.worst_case_sets(pLt, Q["qL_full"]),
            "Tone-set union (ours)": tb.tone_set_union_sets(pLt, pTt, Q["qL_split"], Q["qT"]),
            "Tone-adaptive (ours)": tb.tone_adaptive_sets(pLt, pTt, tau0, lam),
        }
        for m, sets in methods.items():
            cov = sets[np.arange(len(yt)), yt]
            for i, name in enumerate(NAMES):
                mal = (gt == i) & (yt == 1)
                rows.append(dict(method=m, group=name, cancers_caught=cov[mal].mean(),
                                 set_size=sets[gt == i].sum(1).mean()))
    df = pd.DataFrame(rows)
    pT, g = simulate(20000, rng, args.tone_signal)[2:]
    print(f"Tone estimator accuracy: {(pT.argmax(1) == g).mean():.2f}  "
          f"(alpha={args.alpha}, beta={args.beta}, {args.trials} trials)\n")
    out = df.groupby(["method", "group"], sort=False).mean().unstack("group")
    print("Cancers caught per tone (should be >= {:.2f} for valid methods):".format(1 - args.alpha))
    print(out["cancers_caught"][NAMES].round(3).to_string())
    print("\nAverage set size per tone:")
    print(out["set_size"][NAMES].round(3).to_string())


if __name__ == "__main__":
    main()

"""
Tone-blind group-conditional conformal prediction.

Problem
-------
Mondrian (per-skin-tone) conformal prediction guarantees coverage inside every
skin-tone group, but it needs each test patient's skin tone at prediction time.
In real deployments Fitzpatrick type is usually NOT recorded. If you plug in a
*predicted* tone instead, the guarantee is gone: a misclassified patient gets
another group's threshold.

Method: tone-set union
----------------------
Spend the error budget alpha in two parts, alpha = alpha_label + beta.

  1. Tone set T_k(x): a conformal set over skin tones (I-II / III-IV / V-VI),
     calibrated per (tone, label) cell at level beta.
  2. Label thresholds q_label[g, k], calibrated per (tone, label) cell at
     level alpha_label, exactly as in the oracle tone x label method.
  3. Candidate label k goes into the prediction set if it passes the label
     threshold of AT LEAST ONE tone in T_k(x).

Guarantee (union bound). For any true cell (tone g, label k):
    P(miss | G=g, Y=k) <= P(g not in T_k(x) | G=g, Y=k)        <= beta
                        + P(label score > q_label[g,k] | G=g, Y=k) <= alpha_label
                        = alpha
so coverage >= 1 - alpha holds in EVERY skin tone x label cell, e.g. "at least
90% of melanomas caught in FST V-VI", with skin tone never observed at test time.

Concrete trace (2 tones, 1 label, alpha = 0.10, beta = 0.05):
    q_label for light skin = 0.30, for dark skin = 0.55   (dark is harder)
    A test image has malignancy probability 0.52 -> label score 1 - 0.52 = 0.48.
    Tone model says P(light) = 0.97, P(dark) = 0.03, tone threshold 0.90:
        light: 1 - 0.97 = 0.03 <= 0.90 -> in tone set
        dark:  1 - 0.03 = 0.97 >  0.90 -> excluded
    Only light's threshold applies: 0.48 > 0.30 -> "malignant" NOT in the set.
    If instead P(dark) = 0.20, dark stays in the tone set, 0.48 <= 0.55 ->
    "malignant" IS added. Uncertainty about tone widens the set exactly as much
    as needed to protect the tone it might be.

Two reference points bracket the method:
  * plug-in   : use argmax predicted tone, full alpha. Small sets, NO guarantee.
  * worst-case: use every tone (T = all), full alpha. Valid, largest sets.
The tone-set union sits between them; how close to plug-in it gets depends on
how well skin tone can be predicted from the image.
"""

import numpy as np
from scipy.stats import betabinom

from conformal import conformal_quantile


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def cell_quantiles(scores, groups, y, n_groups, n_classes, alpha, cells="tone_label"):
    """
    Conformal threshold for each (group, class) cell -> array [G, K].
    cells="tone"       : one threshold per group, copied across classes
    cells="tone_label" : one threshold per (group, true class) cell
    Empty / too-small cells get +inf (include everything), which keeps validity.
    """
    q = np.full((n_groups, n_classes), np.inf)
    for g in range(n_groups):
        if cells == "tone":
            q[g, :] = conformal_quantile(scores[groups == g], alpha)
        else:
            for k in range(n_classes):
                q[g, k] = conformal_quantile(scores[(groups == g) & (y == k)], alpha)
    return q


def label_scores(p_label, y):
    """LAC score of the true label: 1 - p(y | x)."""
    return 1.0 - p_label[np.arange(len(y)), y]


def tone_scores(p_tone, g):
    """LAC score of the true tone: 1 - p(g | x)."""
    return 1.0 - p_tone[np.arange(len(g)), g]


def calibrate(p_label_cal, y_cal, p_tone_cal, g_cal, alpha, beta, cells="tone_label"):
    """
    Returns everything every method needs, from ONE calibration set:
      qL_full  [G, K]  label thresholds at alpha        (oracle, plug-in, worst-case)
      qL_split [G, K]  label thresholds at alpha - beta (tone-set union)
      qT       [G, K]  tone thresholds at beta          (tone-set union)
    Using the same calibration points for qL_split and qT is fine: the union
    bound only needs each piece to be valid on its own.
    """
    G, K = p_tone_cal.shape[1], p_label_cal.shape[1]
    sL = label_scores(p_label_cal, y_cal)
    sT = tone_scores(p_tone_cal, g_cal)
    return dict(
        qL_full=cell_quantiles(sL, g_cal, y_cal, G, K, alpha, cells),
        qL_split=cell_quantiles(sL, g_cal, y_cal, G, K, alpha - beta, cells),
        qT=cell_quantiles(sT, g_cal, y_cal, G, K, beta, cells),
    )


# ---------------------------------------------------------------------------
# Prediction sets  (all return boolean [n, K])
# ---------------------------------------------------------------------------
def oracle_sets(p_label, g_true, qL):
    """True tone known at test time (the usual Mondrian setting). Reference only."""
    return (1.0 - p_label) <= qL[g_true]


def plugin_sets(p_label, p_tone, qL):
    """Use the predicted tone as if it were true. No guarantee."""
    return (1.0 - p_label) <= qL[p_tone.argmax(1)]


def worst_case_sets(p_label, qL):
    """Ignore tone: the most permissive tone's threshold for each class. Valid."""
    return (1.0 - p_label) <= qL.max(axis=0)[None, :]


def tone_set_union_sets(p_label, p_tone, qL_split, qT, return_tone_sets=False):
    """
    Class k is in the set if some tone g passes BOTH
        tone check : 1 - p_tone[g]  <= qT[g, k]
        label check: 1 - p_label[k] <= qL_split[g, k]
    """
    sT = 1.0 - p_tone                                    # [n, G]
    sL = 1.0 - p_label                                   # [n, K]
    tone_ok = sT[:, :, None] <= qT[None, :, :]           # [n, G, K]
    label_ok = sL[:, None, :] <= qL_split[None, :, :]    # [n, G, K]
    sets = (tone_ok & label_ok).any(axis=1)              # [n, K]
    if return_tone_sets:
        return sets, tone_ok
    return sets


def soft_threshold_sets(p_label, p_tone, qL):
    """
    Heuristic baseline: threshold = tone-probability-weighted average of the
    per-tone thresholds. Smooth, intuitive, NO guarantee. Infinite thresholds
    are capped at 1 (score range of LAC).
    """
    q = np.minimum(qL, 1.0)
    return (1.0 - p_label) <= p_tone @ q


# ---------------------------------------------------------------------------
# Tone-adaptive score + worst-case calibration  (no budget split)
# ---------------------------------------------------------------------------
# Idea: the worst-case union is valid for ANY score function, because for the
# true cell (g, k) the threshold max_g' q[g', k] >= q[g, k]. It is only wasteful
# when the per-tone thresholds differ a lot. So first rescale the score by the
# tone the image probably has, until all tones need about the same threshold;
# then the max over tones costs almost nothing.
#
#   Part A (fit):        tau0[g, k] = per-cell LAC threshold at alpha
#                        (needs label scores + TRUE tone only, no tone model)
#   Expected threshold:  tau(x, k) = sum_g p_tone(g | x) * tau0[g, k]
#   Score:               S(x, k)   = (1 - p_k(x)) / tau(x, k)
#   Part B (calibrate):  lam[g, k] = per-cell conformal quantile of S at alpha
#   Set:                 k in C(x) iff S(x, k) <= max_g lam[g, k]
#
# Trace: tau0 light = 0.30, dark = 0.55. Image with P(light) = 0.9 -> tau = 0.325.
# Label score 0.48 -> S = 1.48. If calibration gives lam = 1.10 for light and
# 1.25 for dark, max = 1.25 < 1.48 -> not included. With P(dark) = 0.5,
# tau = 0.425, S = 1.13 <= 1.25 -> included.
# Part A must be data NOT used in part B. With --tone_train ddi_split the tone
# model's training slice can be reused as part A (it never touches p_tone there).

def tone_adaptive_fit(p_label_A, y_A, g_A, n_groups, alpha, cells="tone_label", shrink=10):
    """
    Per-cell scale tau0 [G, K] for the score. This is only a rescaling, so ANY
    fixed choice keeps the guarantee (validity comes from part B). We therefore
    use a plain empirical (1 - alpha) quantile, which is finite for small cells,
    shrunk toward the pooled per-class quantile:
        tau0[g, k] = (n_gk * q_gk + shrink * q_k) / (n_gk + shrink)
    With ~10 cancers per tone in part A this keeps tau0 from being pure noise.
    """
    s = label_scores(p_label_A, y_A)
    K = p_label_A.shape[1]
    tau0 = np.ones((n_groups, K))
    for k in range(K):
        mk = y_A == k if cells == "tone_label" else np.ones(len(y_A), bool)
        q_k = np.quantile(s[mk], 1 - alpha) if mk.any() else 1.0
        for g in range(n_groups):
            m = mk & (g_A == g)
            n = int(m.sum())
            q_gk = np.quantile(s[m], 1 - alpha) if n else q_k
            tau0[g, k] = (n * q_gk + shrink * q_k) / (n + shrink)
    return np.clip(tau0, 1e-3, 1.0)


def _adaptive_scores(p_label, p_tone, tau0):
    tau = p_tone @ tau0                     # [n, K]
    return (1.0 - p_label) / tau            # [n, K]


def tone_adaptive_calibrate(p_label_B, y_B, p_tone_B, g_B, tau0, alpha, cells="tone_label"):
    S = _adaptive_scores(p_label_B, p_tone_B, tau0)
    s_true = S[np.arange(len(y_B)), y_B]
    return cell_quantiles(s_true, g_B, y_B, tau0.shape[0], tau0.shape[1], alpha, cells)


def tone_adaptive_sets(p_label, p_tone, tau0, lam):
    return _adaptive_scores(p_label, p_tone, tau0) <= lam.max(axis=0)[None, :]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def singleton_rate(sets):
    """Share of predictions that commit to exactly one class (informative sets)."""
    return (sets.sum(1) == 1).astype(float)


def expected_below_target(n_cal, n_test, alpha):
    """
    Probability that an EXACTLY valid split-conformal method still shows
    empirical test coverage below 1 - alpha, purely from sampling noise.

    With n_cal calibration points, coverage given the calibration draw is
    Beta(r, n_cal + 1 - r), r = ceil((n_cal + 1)(1 - alpha)). Test coverage is
    then Beta-Binomial over n_test points.

    Example: n_cal = 50, n_test = 50, alpha = 0.1 -> about 0.30-0.40.
    So "below target in 40% of trials" is what a CORRECT method looks like;
    compare observed rates against this number, not against 0%.
    """
    n_cal, n_test = int(n_cal), int(n_test)
    if n_cal <= 0 or n_test <= 0:
        return np.nan
    r = int(np.ceil((n_cal + 1) * (1 - alpha)))
    if r > n_cal:          # threshold is +inf -> coverage is always 1
        return 0.0
    # count of covered test points < n_test * (1 - alpha)
    cutoff = int(np.ceil(n_test * (1 - alpha))) - 1
    return float(betabinom.cdf(cutoff, n_test, r, n_cal + 1 - r))


def diagnosis_weights(tone, disease, min_per_tone=2):
    """
    Importance weights that give every skin-tone group the SAME diagnosis mix,
    using only diagnoses seen at least `min_per_tone` times in every group.
    Images with other diagnoses get weight 0.

    Why: if FST V-VI simply has harder diagnoses, raw per-tone coverage mixes
    'skin tone effect' with 'disease-mix effect'. Reweighted coverage removes
    the second one.
    Weight for image i (tone t, diagnosis d) = P_all(d) / P_t(d),
    both computed over the shared diagnoses.
    """
    tone = np.asarray(tone)
    disease = np.asarray(disease).astype(str)
    groups = np.unique(tone)
    shared = None
    for t in groups:
        vals, counts = np.unique(disease[tone == t], return_counts=True)
        ok = set(vals[counts >= min_per_tone])
        shared = ok if shared is None else shared & ok
    shared = sorted(shared or [])
    keep = np.isin(disease, shared)
    w = np.zeros(len(tone))
    if not shared:
        return w, shared
    p_all = {d: np.mean(disease[keep] == d) for d in shared}
    for t in groups:
        m = keep & (tone == t)
        for d in shared:
            md = m & (disease == d)
            p_t = md.sum() / m.sum()
            w[md] = p_all[d] / p_t
    return w, shared

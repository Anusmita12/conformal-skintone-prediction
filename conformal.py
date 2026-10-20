"""
Split conformal prediction utilities.

All functions take softmax probabilities `probs` of shape (n, K) and integer
labels `y` of shape (n,).  Prediction sets are returned as boolean arrays of
shape (n, K), where True means "class k is in the set".

Methods
-------
- LAC  (Least Ambiguous set-valued Classifier, Sadinle et al. 2019):
       score = 1 - p_y. Gives the smallest sets on average.
- APS  (Adaptive Prediction Sets, Romano et al. 2020):
       score = cumulative probability mass up to and including the true class
       (randomised). Gives sets that adapt better to hard inputs.
- Mondrian / group-conditional (Vovk 2005): calibrate a separate threshold
       for each group, so coverage holds *within* every group, not only on
       average. A group can be a sensitive attribute (skin tone) or the true
       class itself ("class-conditional" conformal).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Quantile
# ---------------------------------------------------------------------------
def conformal_quantile(scores, alpha):
    """
    Finite-sample conformal threshold: the r-th smallest calibration score,
    r = ceil((n + 1)(1 - alpha)). Guarantees coverage >= 1 - alpha.

    Taken as an exact order statistic. (np.quantile(..., method="higher") at
    level r/n is off by one rank and over-covers badly for small n, e.g.
    ~95% instead of ~90.5% at n = 20.)
    """
    scores = np.asarray(scores)
    n = len(scores)
    r = int(np.ceil((n + 1) * (1 - alpha)))
    if n == 0 or r > n:    # too few points for this alpha: the set must be everything
        return np.inf
    return np.sort(scores)[r - 1]


# ---------------------------------------------------------------------------
# LAC
# ---------------------------------------------------------------------------
def lac_scores(probs, y):
    return 1.0 - probs[np.arange(len(y)), y]


def lac_sets(probs, qhat):
    return probs >= 1.0 - qhat


# ---------------------------------------------------------------------------
# APS (randomised)
# ---------------------------------------------------------------------------
def aps_scores(probs, y, rng):
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    rank_of_y = np.argmax(order == y[:, None], axis=1)
    idx = np.arange(len(y))
    u = rng.random(len(y))
    # mass strictly above y, plus a random fraction of p_y
    return cumsum[idx, rank_of_y] - u * sorted_p[idx, rank_of_y]


def aps_sets(probs, qhat, rng):
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    u = rng.random((probs.shape[0], 1))
    # include class at rank r if (mass above it) + u * p_r <= qhat
    include_sorted = (cumsum - sorted_p) + u * sorted_p <= qhat
    include_sorted[:, 0] = True            # never return an empty set
    sets = np.zeros_like(include_sorted)
    np.put_along_axis(sets, order, include_sorted, axis=1)
    return sets


# ---------------------------------------------------------------------------
# Group-conditional (Mondrian) LAC
# ---------------------------------------------------------------------------
def mondrian_lac_qhats(probs, y, groups, alpha):
    """One LAC threshold per group value."""
    scores = lac_scores(probs, y)
    return {g: conformal_quantile(scores[groups == g], alpha)
            for g in np.unique(groups)}


def mondrian_lac_sets(probs, groups, qhats):
    """Groups are known at test time (e.g. skin tone recorded in metadata)."""
    sets = np.zeros(probs.shape, dtype=bool)
    for g, q in qhats.items():
        m = groups == g
        sets[m] = lac_sets(probs[m], q)
    return sets


def class_conditional_lac_sets(probs, qhats):
    """
    Class-conditional conformal: the 'group' is the unknown true label, so
    each candidate class k is tested against its own threshold qhat_k.
    Guarantees coverage separately for every true class.
    """
    K = probs.shape[1]
    q = np.array([qhats.get(k, np.inf) for k in range(K)])
    return probs >= 1.0 - q[None, :]


def group_label_lac_qhats(probs, y, groups, alpha):
    """One LAC threshold per (group, true label) cell."""
    scores = lac_scores(probs, y)
    return {(g, k): conformal_quantile(scores[(groups == g) & (y == k)], alpha)
            for g in np.unique(groups) for k in range(probs.shape[1])}


def group_label_lac_sets(probs, groups, qhats):
    """
    Group x label conditional conformal: within each group (known at test
    time), candidate class k is tested against the threshold calibrated on
    that group's examples whose true label is k. Guarantees coverage for every
    (group, true label) cell, e.g. >= 90% of cancers caught in EACH skin tone.
    """
    sets = np.zeros(probs.shape, dtype=bool)
    for g in np.unique(groups):
        m = groups == g
        q = np.array([qhats.get((g, k), np.inf) for k in range(probs.shape[1])])
        sets[m] = probs[m] >= 1.0 - q[None, :]
    return sets


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def coverage(sets, y):
    return sets[np.arange(len(y)), y].astype(float)


def set_size(sets):
    return sets.sum(axis=1).astype(float)
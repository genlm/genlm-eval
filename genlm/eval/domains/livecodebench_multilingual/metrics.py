"""pass@k for multilingual LiveCodeBench.

genlm-eval's native metric is weighted accuracy (computed by ``evaluate_ensemble``); pass@k
is added for comparability with the LiveCodeBench / Multi-LCB leaderboards. ``pass_at_k`` is
the standard unbiased estimator (Chen et al., 2021): given ``n`` independent samples of which
``c`` are correct, the probability that at least one of ``k`` drawn samples is correct.
"""

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimate for one problem: n samples, c correct, draw k (assumes k <= n)."""
    if n < 1:
        raise ValueError("n must be >= 1 (no samples to estimate pass@k)")
    if k <= 0:
        raise ValueError("k must be >= 1")
    if c < 0 or c > n:
        raise ValueError("require 0 <= c <= n")
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def pass_at_k_from_scores(per_sample_scores, k: int) -> float:
    """pass@k from one problem's per-sample 0/1 scores (e.g. EvaluationResult ``results``)."""
    scores = list(per_sample_scores)
    n = len(scores)
    c = sum(1 for s in scores if s > 0)
    return pass_at_k(n, c, k)

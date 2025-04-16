import numpy as np


def bootstrap_ci(values, metric, ci=0.95, n_bootstrap=10000):
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")

    if not values:
        raise ValueError("values must not be empty")

    values = np.asarray(values)

    mean = metric(values)
    bootstrap_values = []

    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_values.append(metric(bootstrap_sample))

    lower_percentile = (1 - ci) / 2 * 100
    upper_percentile = (1 + ci) / 2 * 100
    lower, upper = np.percentile(bootstrap_values, [lower_percentile, upper_percentile])
    return mean, lower, upper

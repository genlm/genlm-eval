import numpy as np
from abc import abstractmethod, ABC
from collections import OrderedDict


def bootstrap_ci(values, metric, ci=0.95, n_bootstrap=10000):
    """Calculate bootstrap confidence intervals for a given metric.

    Args:
        values (array-like): Array-like object containing the data points.
        metric (function): Function that computes the statistic of interest (e.g., np.mean, np.median).
        ci (float): Confidence interval level (between 0 and 1). Default is 0.95 for 95% CI.
        n_bootstrap (int): Number of bootstrap samples to generate. Default is 10000.

    Returns:
        (tuple): (mean, lower, upper) where:
            - mean is the metric computed on the original data
            - lower is the lower bound of the confidence interval
            - upper is the upper bound of the confidence interval

    Raises:
        ValueError: If ci is not between 0 and 1 or if values is empty.
    """
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


def chat_template_messages(prompt, examples, user_message):
    """Create a list of messages for chat template formatting.

    Args:
        prompt (str): System prompt to be used as the initial message.
        examples (list): List of (example, response) tuples for few-shot learning.
        user_message (str): The actual user query to be processed.

    Returns:
        (list): List of dictionaries containing the formatted messages with roles
             and content for the chat template.
    """
    messages = [{"role": "system", "content": prompt}]
    for example, response in examples:
        messages.append({"role": "user", "content": example})
        messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": user_message})
    return messages


class LRUCache(ABC):
    """A cache that evicts the least recently used item when the cache is full."""

    def __init__(self, cache_size=1):
        """Initialize the cache.

        Args:
            cache_size (int): The size of the cache.
        """
        self.cache_size = cache_size
        self.cache = OrderedDict()

    @abstractmethod
    def create(self, key):
        """Create an object for a given key.

        Args:
            key (any): The key to create an object for.
        """
        pass  # pragma: no cover

    def cleanup(self, key, obj):
        """Cleanup an object for a given key.

        Args:
            key (any): The key to cleanup an object for.
            obj (any): The object to cleanup.
        """
        pass  # pragma: no cover

    def get(self, key):
        """Get an object for a given key.

        Args:
            key (any): The key to get an object for.

        Returns:
            any: The object for the given key.
        """
        if self.cache_size > 0:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        else:
            return self.create(key)

        obj = self.create(key)
        self.cache[key] = obj

        if len(self.cache) > self.cache_size:
            key, obj = self.cache.popitem(last=False)
            self.cleanup(key, obj)

        return obj

import numpy as np
from abc import abstractmethod, ABC
from collections import OrderedDict


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


def chat_template_messages(prompt, examples, user_message):
    messages = [{"role": "system", "content": prompt}]
    for example, response in examples:
        messages.append({"role": "user", "content": example})
        messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": user_message})
    return messages


class LRUCache(ABC):
    """A cache that evicts the least recently used item when the cache is full."""

    def __init__(self, cache_size=1):
        self.cache_size = cache_size
        self.cache = OrderedDict()

    @abstractmethod
    def create(self, key):
        pass  # pragma: no cover

    def cleanup(self, key, obj):
        pass  # pragma: no cover

    def get(self, key):
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

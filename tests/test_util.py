import pytest
import numpy as np
from genlm.eval.util import bootstrap_ci, chat_template_messages, LRUCache


def test_bootstrap_ci_basic():
    # Test with simple mean calculation
    values = [1, 2, 3, 4, 5]
    mean, lower, upper = bootstrap_ci(values, np.mean, ci=0.95, n_bootstrap=1000)

    assert isinstance(mean, float)
    assert isinstance(lower, float)
    assert isinstance(upper, float)
    assert lower <= mean <= upper
    assert abs(mean - 3.0) < 0.1  # Mean should be close to 3


def test_bootstrap_ci_different_metrics():
    values = [1, 2, 3, 4, 5]

    # Test with median
    mean, _, _ = bootstrap_ci(values, np.median, ci=0.95, n_bootstrap=1000)
    assert abs(mean - 3.0) < 0.1  # Median should be close to 3

    # Test with standard deviation
    mean, _, _ = bootstrap_ci(values, np.std, ci=0.95, n_bootstrap=1000)
    assert 1.0 < mean < 2.0  # std should be around 1.58


def test_bootstrap_ci_invalid_ci():
    values = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError, match="ci must be between 0 and 1"):
        bootstrap_ci(values, np.mean, ci=1.5)


def test_bootstrap_ci_different_confidence_levels():
    values = [1, 2, 3, 4, 5]

    # Test with 90% confidence interval
    mean90, lower90, upper90 = bootstrap_ci(values, np.mean, ci=0.90, n_bootstrap=1000)

    # Test with 99% confidence interval
    mean99, lower99, upper99 = bootstrap_ci(values, np.mean, ci=0.99, n_bootstrap=1000)

    # 99% CI should be wider than 90% CI
    assert (upper99 - lower99) > (upper90 - lower90)
    # Means should be approximately equal
    assert abs(mean90 - mean99) < 0.1


def test_bootstrap_ci_empty_array():
    with pytest.raises(
        Exception
    ):  # Either ValueError or IndexError depending on metric
        bootstrap_ci([], np.mean)


def test_bootstrap_ci_single_value():
    values = [42]
    mean, lower, upper = bootstrap_ci(values, np.mean, ci=0.95, n_bootstrap=1000)
    assert mean == 42
    assert lower == 42
    assert upper == 42


def test_chat_template_messages():
    prompt = "You are a helpful assistant."
    examples = [
        ("Hello, how are you?", "Hello, how are you?"),
        ("What is the capital of France?", "Paris"),
    ]
    user_message = "What is the capital of the moon?"
    messages = chat_template_messages(prompt, examples, user_message)

    assert len(messages) == 2 * len(examples) + 2
    assert messages[0]["role"] == "system"

    for i in range(1, len(messages) - 1, 2):
        assert messages[i]["role"] == "user"
        assert messages[i + 1]["role"] == "assistant"

    assert messages[0]["content"] == prompt

    for i, (example, response) in enumerate(examples):
        assert messages[2 * i + 1]["content"] == example
        assert messages[2 * i + 2]["content"] == response

    assert messages[-1]["content"] == user_message


def test_lru_cache_basic():
    class TestCache(LRUCache):
        def create(self, key):
            return f"value_{key}"

    cache = TestCache(cache_size=2)

    # Test basic cache functionality
    assert cache.get("a") == "value_a"
    assert cache.get("a") == "value_a"  # Should hit cache
    assert len(cache.cache) == 1

    # Test cache eviction
    cache.get("b")
    cache.get("c")  # This should evict 'a'
    assert "a" not in cache.cache
    assert "b" in cache.cache
    assert "c" in cache.cache
    assert len(cache.cache) == 2


def test_lru_cache_zero_size():
    class TestCache(LRUCache):
        def create(self, key):
            return f"value_{key}"

    cache = TestCache(cache_size=0)

    # Should always create new value
    assert cache.get("a") == "value_a"
    assert len(cache.cache) == 0


def test_lru_cache_cleanup():
    cleaned_items = []

    class TestCache(LRUCache):
        def create(self, key):
            return f"value_{key}"

        def cleanup(self, key, obj):
            cleaned_items.append((key, obj))

    cache = TestCache(cache_size=1)

    cache.get("a")
    cache.get("b")  # Should evict 'a'

    assert len(cleaned_items) == 1
    assert cleaned_items[0] == ("a", "value_a")

"""
Unit tests for dataset sampling and preparation utilities.
"""

import numpy as np
import pytest
from sqlllama.data.prepare_dataset import select_string_increasing_prob


class TestSelectStringIncreasingProb:
    """Verifies length-weighted sampling behavior and constraints."""

    def test_samples_exact_number_of_items(self):
        pool = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff", "ggggggg"]
        sampled = select_string_increasing_prob(pool, numb_items=4)
        assert len(sampled) == 4
        assert len(set(sampled)) == 4
        assert all(0 <= idx < len(pool) for idx in sampled)

    def test_raises_on_request_exceeding_pool_size(self):
        pool = ["a", "b", "c"]
        with pytest.raises(ValueError, match="Requested 5 items"):
            select_string_increasing_prob(pool, numb_items=5)

    def test_sampling_bias_towards_longer_strings(self):
        # 95 very short strings and 5 very long strings
        np.random.seed(42)
        short_strings = ["x"] * 95
        long_strings = ["x" * 200] * 5
        pool = short_strings + long_strings

        # Sample 5 items multiple times and check if long strings dominate
        sampled = select_string_increasing_prob(pool, numb_items=5, power=4.0)
        # The long string indices are 95, 96, 97, 98, 99
        long_indices_sampled = [idx for idx in sampled if idx >= 95]
        assert len(long_indices_sampled) >= 3

    def test_all_equal_length_strings(self):
        pool = ["same"] * 10
        sampled = select_string_increasing_prob(pool, numb_items=5)
        assert len(sampled) == 5
        assert len(set(sampled)) == 5

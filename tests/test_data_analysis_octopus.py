"""Unit tests for data_analysis_octopus's pure outlier/stats helpers.

Covers detect_outliers_iqr, transform_outliers, process_outliers,
count_percentage, create_feature_dataframe, and get_information_value (see
issue #21). These are the module's pure(-ish) functions -- plain
pandas/numpy in and out, no plotting or IPython side effects -- which makes
them testable in isolation, unlike DataViz or the sklearn-model-training
helpers.

Note on the IQR-boundary scenario: a value already equal to the clip bound
is behaviorally indistinguishable, via the returned data, from a value that
was clipped to that same bound -- clipping a value to itself is a no-op
either way, so "value exactly at the bound" can't be asserted on its own.
What's actually observable and meaningful is the contrast this suite
checks: values strictly inside the bounds are left alone, values outside
are clipped to the bound.
"""

import numpy as np
import pandas as pd
import pytest

import data_analysis_octopus as dao


def test_process_outliers_clips_values_above_upper_bound():
    data = pd.DataFrame({"x": [10, 11, 12, 11, 10, 12, 11, 1000]})
    _, upper_bound = dao.detect_outliers_iqr(data, "x")

    result = dao.process_outliers(data.copy(), "x")

    assert result["x"].max() == upper_bound
    assert 1000 not in result["x"].values


def test_process_outliers_leaves_values_within_bounds_unchanged():
    data = pd.DataFrame({"x": [10, 11, 12, 11, 10, 12, 11, 1000]})
    original_inliers = data["x"].iloc[:-1].tolist()

    result = dao.process_outliers(data.copy(), "x")

    assert result["x"].iloc[:-1].tolist() == original_inliers


def test_detect_outliers_iqr_handles_zero_variance_column_without_raising():
    data = pd.DataFrame({"x": [5, 5, 5, 5, 5]})

    lower_bound, upper_bound = dao.detect_outliers_iqr(data, "x")

    assert lower_bound == 5
    assert upper_bound == 5


def test_count_percentage_sums_to_100_with_a_single_unique_value():
    data = pd.DataFrame({"category": ["a", "a", "a"]})

    result = dao.count_percentage(data, "category")

    assert result["porcentaje"].sum() == 100


def test_count_percentage_sums_to_100_with_multiple_unique_values():
    data = pd.DataFrame({"category": ["a", "a", "b", "c", "c", "c", "d"]})

    result = dao.count_percentage(data, "category")

    # rounding to 2 decimals per-category (see count_percentage) can leave
    # the sum a few hundredths off 100; that's within tolerance.
    assert result["porcentaje"].sum() == pytest.approx(100, abs=0.05)


def test_create_feature_dataframe_uses_only_first_row():
    counted = pd.DataFrame(
        {
            "category": ["frequent_value", "other_value"],
            "conteo": [95, 5],
            "porcentaje": [95.0, 5.0],
        }
    )

    result = dao.create_feature_dataframe(counted, "category")

    assert len(result) == 1
    assert result.loc[0, "category"] == "frequent_value"
    assert result.loc[0, "conteo"] == 95
    assert result.loc[0, "porcentaje"] == 95.0


def test_get_information_value_does_not_raise_with_zero_event_category():
    data = pd.DataFrame(
        {
            "category": ["a", "a", "b", "b", "b"],
            "target": [1, 1, 0, 0, 0],
        }
    )

    iv = dao.get_information_value(data, "category", "target")

    assert isinstance(iv, (float, np.floating))

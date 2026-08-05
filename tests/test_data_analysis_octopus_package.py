"""Regression tests for the DataViz extraction (see issue #22).

data_analysis_octopus was split into a package (__init__.py + viz.py) with
DataViz moved into viz.py and re-exported from __init__.py, so every
existing notebook import style keeps resolving to the same class.
"""

import data_analysis_octopus as dao
from data_analysis_octopus import viz

DATAVIZ_METHODS = [
    "completitud",
    "histograma",
    "caja",
    "barras_horizontales",
    "lineas",
    "puntos",
    "densidad",
    "violin",
    "calor",
]


def test_dataviz_importable_from_its_own_submodule():
    assert hasattr(viz, "DataViz")
    for method in DATAVIZ_METHODS:
        assert hasattr(viz.DataViz, method)


def test_namespaced_import_resolves_to_the_same_dataviz_class():
    assert dao.DataViz is viz.DataViz


def test_wildcard_import_still_exposes_dataviz():
    namespace = {}
    exec("from data_analysis_octopus import *", namespace)  # pylint: disable=exec-used
    assert namespace["DataViz"] is viz.DataViz


def test_non_plotting_helpers_still_import_and_are_unaffected_by_the_split():
    assert callable(dao.detect_outliers_iqr)
    assert callable(dao.count_percentage)
    assert callable(dao.train_classifier_model)

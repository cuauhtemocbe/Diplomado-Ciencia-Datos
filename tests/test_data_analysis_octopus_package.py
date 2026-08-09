"""Regression tests for the DataViz extraction (see issue #22) and the
heatmap/plot_heatmap_clusters move (see issue #34).

data_analysis_octopus was split into a package (__init__.py + viz.py) with
DataViz moved into viz.py and re-exported from __init__.py, so every
existing notebook import style keeps resolving to the same class. heatmap
and plot_heatmap_clusters -- visualization code left behind in __init__.py
after the DataViz split -- were later moved into viz.py the same way, since
notebooks/15-AirBnb.ipynb still calls them via dao.heatmap/dao.plot_heatmap_clusters.
"""

import pandas as pd

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


def test_heatmap_and_plot_heatmap_clusters_importable_from_their_own_submodule():
    assert hasattr(viz, "heatmap")
    assert hasattr(viz, "plot_heatmap_clusters")


def test_namespaced_import_resolves_heatmap_functions_to_the_viz_submodule():
    assert dao.heatmap is viz.heatmap
    assert dao.plot_heatmap_clusters is viz.plot_heatmap_clusters


def test_densidad_preserves_no_hue_call_shape(monkeypatch):
    calls = []
    monkeypatch.setattr(viz.sns, "kdeplot", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(viz.plt, "show", lambda: None)

    dao.DataViz.densidad(pd.DataFrame({"value": [1, 2, 3]}), ["value"])

    assert len(calls) == 1
    assert calls[0]["x"] == "value"
    assert calls[0]["data"]["value"].tolist() == [1, 2, 3]


def test_densidad_passes_hue_to_seaborn(monkeypatch):
    calls = []
    monkeypatch.setattr(viz.sns, "kdeplot", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(viz.plt, "show", lambda: None)
    data = pd.DataFrame({"value": [1, 2, 3, 4], "target": [0, 1, 0, 1]})

    dao.DataViz.densidad(data, ["value"], hue="target")

    assert calls == [{"data": data, "x": "value", "hue": "target"}]

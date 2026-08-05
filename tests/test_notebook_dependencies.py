"""Per-notebook import smoke tests.

Each entry lists the non-stdlib top-level modules a notebook imports
directly (see notebooks/<name>.ipynb). A module missing in the current
environment means that notebook's Poetry group isn't installed here, so the
test is skipped rather than failed -- this lets the suite run in any
container (core-only, or with a subset of --with groups) without needing
every dependency group installed everywhere.

Rolling out incrementally: only the first 5 notebooks (by filename order)
are covered so far.
"""

import pytest

NOTEBOOK_DEPENDENCIES = [
    ("0-Hello-Pandas", ["pandas"]),
    ("1-Ecobici-análisis", ["pandas", "bs4"]),
    ("2-Cardiotocography", ["pandas", "data_analysis_octopus"]),
    ("3-Starbucks", ["ipywidgets", "scipy", "numpy", "pandas", "plotly", "seaborn"]),
    (
        "4-Restaurantes",
        ["ipywidgets", "data_analysis_octopus", "sklearn", "matplotlib", "numpy", "pandas", "plotly"],
    ),
]


@pytest.mark.parametrize("notebook,modules", NOTEBOOK_DEPENDENCIES, ids=[n for n, _ in NOTEBOOK_DEPENDENCIES])
def test_notebook_dependencies_importable(notebook, modules):
    for module in modules:
        pytest.importorskip(module, reason=f"{notebook} needs '{module}', not installed in this environment")

"""Per-notebook import smoke tests.

Each entry lists the non-stdlib top-level modules a notebook imports
directly (see notebooks/<name>.ipynb). A module missing in the current
environment means that notebook's Poetry group isn't installed here, so the
test is skipped rather than failed -- this lets the suite run in any
container (core-only, or with a subset of --with groups) without needing
every dependency group installed everywhere.

Two entries are permanently skip-only, not tied to any Poetry group:
- "Predicción_precios_casas_CNN" imports `cv2` (opencv-python), which isn't
  a declared dependency in any group -- always skips.
- "Predicción_precios_casas_CNN" also imports `google.colab`, a module that
  only exists inside the Colab runtime and can't be installed here -- always
  skips.
"""

from pathlib import Path

import pytest

NOTEBOOK_DEPENDENCIES = [
    ("0-Hello-Pandas", ["pandas"]),
    ("1-Ecobici-análisis", ["data_analysis_octopus", "pandas", "bs4"]),
    ("2-Cardiotocography", ["pandas", "data_analysis_octopus"]),
    ("3-Starbucks", ["data_analysis_octopus", "ipywidgets", "scipy", "numpy", "pandas", "plotly", "seaborn"]),
    (
        "4-Restaurantes",
        [
            "ipywidgets",
            "data_analysis_octopus",
            "sklearn",
            "matplotlib",
            "numpy",
            "pandas",
            "plotly",
        ],
    ),
    (
        "5-Regresiones",
        [
            "cufflinks",
            "data_analysis_octopus",
            "numpy",
            "pandas",
            "sklearn",
            "unidecode",
        ],
    ),
    (
        "6-Whatsapp",
        ["data_analysis_octopus", "nltk", "pandas", "sklearn", "unidecode", "xgboost"],
    ),
    (
        "7-Clasificación-clientes",
        ["data_analysis_octopus", "numpy", "pandas", "scipy", "sklearn"],
    ),
    (
        "8-Clasificación-cáncer",
        [
            "data_analysis_octopus",
            "matplotlib",
            "numpy",
            "pandas",
            "plotly",
            "seaborn",
            "sklearn",
        ],
    ),
    (
        "9-Electrocardiograma",
        [
            "IPython",
            "data_analysis_octopus",
            "joblib",
            "matplotlib",
            "neurokit2",
            "numpy",
            "pandas",
            "scipy",
            "seaborn",
            "sklearn",
            "wfdb",
            "xgboost",
        ],
    ),
    (
        "10-Proyecto-Hipertension-Mexico",
        [
            "cufflinks",
            "data_analysis_octopus",
            "matplotlib",
            "numpy",
            "pandas",
            "plotly",
            "scipy",
            "seaborn",
            "shap",
            "sklearn",
            "xgboost",
        ],
    ),
    (
        "11-Indice-de-marginalidad",
        [
            "data_analysis_octopus",
            "geopandas",
            "numpy",
            "pandas",
            "plotly",
            "seaborn",
            "sklearn",
        ],
    ),
    (
        "12-Ataque-corazon",
        [
            "data_analysis_octopus",
            "numpy",
            "pandas",
            "plotly",
            "scipy",
            "seaborn",
            "sklearn",
        ],
    ),
    (
        "13-Agrupamiento-texto",
        ["app_clustering", "dotenv", "numpy", "pandas", "plotly", "sklearn", "umap"],
    ),
    (
        "14-FIFA",
        [
            "matplotlib",
            "numpy",
            "pandas",
            "phik",
            "plotly",
            "scipy",
            "seaborn",
            "sklearn",
        ],
    ),
    (
        "15-AirBnb",
        [
            "data_analysis_octopus",
            "matplotlib",
            "numpy",
            "pandas",
            "phik",
            "plotly",
            "scipy",
            "seaborn",
            "sklearn",
            "statsmodels",
            "yellowbrick",
        ],
    ),
    (
        "16-Manuscrita",
        ["cufflinks", "h5py", "keras", "numpy", "pandas", "sklearn", "tensorflow"],
    ),
    (
        "Clasificación_Pokémons_Base",
        [
            "IPython",
            "PIL",
            "matplotlib",
            "numpy",
            "pandas",
            "sklearn",
            "tensorflow",
            "tqdm",
        ],
    ),
    (
        "Clasificación_Pokémons",
        [
            "cufflinks",
            "keras",
            "matplotlib",
            "numpy",
            "pandas",
            "sklearn",
            "tensorflow",
        ],
    ),
    (
        "Predicción_precios_casas_CNN",
        [
            "data_analysis_octopus",
            "IPython",
            "PIL",
            "cv2",
            "google.colab",
            "keras",
            "matplotlib",
            "numpy",
            "pandas",
            "requests",
            "seaborn",
            "sklearn",
            "tensorflow",
            "tqdm",
        ],
    ),
]


@pytest.mark.parametrize(
    "notebook,modules", NOTEBOOK_DEPENDENCIES, ids=[n for n, _ in NOTEBOOK_DEPENDENCIES]
)
def test_notebook_dependencies_importable(notebook, modules):
    for module in modules:
        pytest.importorskip(
            module,
            reason=f"{notebook} needs '{module}', not installed in this environment",
        )


def test_every_notebook_has_a_dependency_entry():
    notebooks_dir = Path(__file__).resolve().parent.parent / "notebooks"
    notebook_names = {path.stem for path in notebooks_dir.glob("*.ipynb")}
    covered_names = {name for name, _ in NOTEBOOK_DEPENDENCIES}
    missing = notebook_names - covered_names
    assert (
        not missing
    ), f"Notebooks missing from NOTEBOOK_DEPENDENCIES: {sorted(missing)}"

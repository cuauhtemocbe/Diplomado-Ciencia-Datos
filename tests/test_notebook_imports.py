"""Import-hygiene tests for notebooks that depend on data_analysis_octopus.

These notebooks were converted from `from data_analysis_octopus import *`
to an explicit `import data_analysis_octopus as dao` (see issue #19).

A wildcard import doesn't just expose the module's own functions/classes --
it also re-exports every name the module itself imports at top level (e.g.
`np`, `pd`, `datetime`, `display`, sklearn classes). Several notebooks
turned out to rely on some of those leaked names directly (bare `np.array`,
bare `display(df)`, etc.) without importing them explicitly themselves.
Converting to a namespaced import removes that leak, so this suite checks
against the *full* set of names data_analysis_octopus binds at module
level (definitions + its own imports), not just the functions/classes it
defines.

Some notebooks also locally redefine (shadow) a subset of these names with
their own version -- in those cells the notebook was never actually
resolving the name via data_analysis_octopus to begin with (Python name
resolution favors the later, local definition in the same execution
order), so a shadowed name is treated as correctly resolved without a
`dao.` prefix or an explicit import, matching the notebook's pre-existing
(unchanged) runtime behavior.
"""

import ast
import io
import json
import re
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest

from tests.test_notebook_dependencies import NOTEBOOK_DEPENDENCIES

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
SRC_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "data_analysis_octopus"
    / "__init__.py"
)

CONVERTED_NOTEBOOKS = [
    "2-Cardiotocography",
    "3-Starbucks",
    "6-Whatsapp",
    "7-Clasificación-clientes",
    "9-Electrocardiograma",
    "10-Proyecto-Hipertension-Mexico",
    "12-Ataque-corazon",
    "15-AirBnb",
]

# Full top-to-bottom execution is verified for these two (matches issue
# #19's Gherkin Examples table). The other three converted notebooks are
# still covered by the static scenarios below; executing them too would
# only add heavier-group runtime cost without exercising new import-hygiene
# behavior.
EXECUTION_VERIFIED_NOTEBOOKS = ["2-Cardiotocography", "9-Electrocardiograma"]

DEPENDENCIES_BY_NOTEBOOK = dict(NOTEBOOK_DEPENDENCIES)


def _module_level_names():
    """Every name data_analysis_octopus.py binds at module level: its own
    defs/classes/assignments, plus everything it imports (all of which
    leaks through `from data_analysis_octopus import *`)."""
    tree = ast.parse(SRC_PATH.read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


MODULE_LEVEL_NAMES = _module_level_names()


def _code_cells(notebook_name):
    nb = json.loads(
        (NOTEBOOKS_DIR / f"{notebook_name}.ipynb").read_text(encoding="utf-8")
    )
    return [
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    ]


def _import_statements(source):
    return [
        line.strip()
        for line in source.splitlines()
        if re.match(r"^\s*(import|from)\s+data_analysis_octopus\b", line)
    ]


def _own_imports(source, names):
    """Names this cell binds via its own (non-data_analysis_octopus)
    import statements -- these satisfy a bare reference just as well as a
    `dao.` prefix would."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        cleaned = "\n".join(
            l for l in source.splitlines() if not l.strip().startswith(("%", "!"))
        )
        try:
            tree = ast.parse(cleaned)
        except SyntaxError:
            return set()
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if "data_analysis_octopus" in a.name:
                    continue
                name = a.asname or a.name.split(".")[0]
                if name in names:
                    found.add(name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and "data_analysis_octopus" in node.module:
                continue
            for a in node.names:
                name = a.asname or a.name
                if name in names:
                    found.add(name)
    return found


def _local_definitions(source, names):
    defs = set()
    for line in source.splitlines():
        m = re.match(r"^\s*(def|class)\s+(\w+)", line)
        if m and m.group(2) in names:
            defs.add(m.group(2))
        m2 = re.match(r"^(\w+)\s*(?::\s*\w+)?\s*=", line)
        if m2 and m2.group(1) in names:
            defs.add(m2.group(1))
    return defs


def _bare_references(source, names):
    bare = set()
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    for i, tok in enumerate(tokens):
        if tok.type == tokenize.NAME and tok.string in names:
            prev = tokens[i - 1] if i > 0 else None
            next_ = tokens[i + 1] if i + 1 < len(tokens) else None
            if prev and prev.type == tokenize.OP and prev.string == ".":
                continue  # attribute access, e.g. dao.DataViz
            if prev and prev.type == tokenize.NAME and prev.string in ("def", "class"):
                continue  # definition site, e.g. def freq_discrete(...)
            if next_ and next_.type == tokenize.OP and next_.string == "=":
                continue  # simple assignment target, e.g. ModelClassifier = Union[...]
            if tok.line.strip().startswith(("import ", "from ")):
                continue
            bare.add(tok.string)
    return bare


@pytest.mark.parametrize("notebook", CONVERTED_NOTEBOOKS)
def test_no_wildcard_import(notebook):
    for cell_source in _code_cells(notebook):
        assert "from data_analysis_octopus import *" not in cell_source


@pytest.mark.parametrize("notebook", CONVERTED_NOTEBOOKS)
def test_at_most_one_import_statement(notebook):
    statements = []
    for cell_source in _code_cells(notebook):
        statements.extend(_import_statements(cell_source))
    assert len(statements) <= 1, (
        f"{notebook} has {len(statements)} data_analysis_octopus import "
        f"statements: {statements}"
    )


@pytest.mark.parametrize("notebook", CONVERTED_NOTEBOOKS)
def test_no_dangling_bare_reference(notebook):
    resolved = set()
    for cell_source in _code_cells(notebook):
        this_cell_resolved = _own_imports(
            cell_source, MODULE_LEVEL_NAMES
        ) | _local_definitions(cell_source, MODULE_LEVEL_NAMES)
        effective = resolved | this_cell_resolved
        bare = _bare_references(cell_source, MODULE_LEVEL_NAMES)
        dangling = bare - effective
        assert not dangling, (
            f"{notebook} has a bare reference to {dangling} that is neither "
            f"dao.-prefixed, explicitly imported, nor shadowed by a local "
            f"definition"
        )
        resolved = effective


# 9-Electrocardiograma reads from data/electrocardiograma/, which has never
# been committed to this repo (confirmed via `git log --all` -- not a file
# this change removed). Executing it end to end is only possible once that
# dataset is provided out of band; skip rather than fail so this pre-existing,
# unrelated gap doesn't masquerade as an import-hygiene regression.
REQUIRED_DATA_DIRS = {
    "9-Electrocardiograma": NOTEBOOKS_DIR.parent / "data" / "electrocardiograma",
}


@pytest.mark.parametrize("notebook", EXECUTION_VERIFIED_NOTEBOOKS)
def test_notebook_executes_end_to_end(notebook, tmp_path):
    for module in DEPENDENCIES_BY_NOTEBOOK.get(notebook, []):
        pytest.importorskip(
            module, reason=f"{notebook} needs '{module}', not installed here"
        )

    required_data_dir = REQUIRED_DATA_DIRS.get(notebook)
    if required_data_dir is not None and not required_data_dir.is_dir():
        pytest.skip(
            f"{notebook} needs {required_data_dir}, not present in this checkout"
        )

    notebook_path = NOTEBOOKS_DIR / f"{notebook}.ipynb"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=900",
            "--output-dir",
            str(tmp_path),
            notebook_path.name,
        ],
        capture_output=True,
        text=True,
        cwd=str(NOTEBOOKS_DIR),
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]

import ast
import glob
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODULES = sorted(glob.glob(os.path.join(ROOT, "tests", "**", "test_*.py"), recursive=True))


@pytest.mark.parametrize("path", MODULES, ids=[os.path.relpath(p, ROOT) for p in MODULES])
def test_no_duplicate_test_functions(path):
    """A second definition with the same name silently replaces the first, so the first never
    runs. Every module-level function name must be unique."""
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), path)
    names = [node.name for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, "redefined in {}: {}".format(os.path.relpath(path, ROOT), duplicates)

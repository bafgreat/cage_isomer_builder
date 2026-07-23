import tempfile
from pathlib import Path

import pytest

from cage_isomer_builder.utils.uff4mof_params import load_uff4mof_db


@pytest.fixture
def tmp_dir():
    """
    A fresh temporary directory per test, as a Path.

    The test files in this directory predate pytest in this repo (see
    their own docstrings) and are written as plain scripts whose test_*
    functions take tmp_dir/db as ordinary parameters, filled in by their
    own __main__ block. Under pytest, any test function parameter is
    resolved as a fixture request instead, so this (and db below) exist
    purely to satisfy that without having to change those functions'
    signatures.
    """
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture(scope="session")
def db():
    """The UFF4MOF parameter database, loaded once per test session."""
    return load_uff4mof_db()

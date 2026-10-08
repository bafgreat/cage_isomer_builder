import tempfile
from pathlib import Path

import pytest

from cage_isomer_builder.utils.uff4mof_params import load_uff4mof_db


@pytest.fixture
def tmp_dir():
    """
    A fresh temporary directory per test, as a Path.

    Some test files can also run as plain scripts, where their __main__
    block passes tmp_dir and db to the test functions. Under pytest these
    parameters are fixtures, provided here and below.
    """
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture(autouse=True)
def _fail_on_recorded_checks(request):
    """
    Script-style test modules record failed checks in a module-level
    FAILURES list instead of raising. Turn any check that failed during a
    test into a real pytest failure, so those tests cannot pass unnoticed.
    """
    failures = getattr(request.module, "FAILURES", None)
    if failures is None:
        yield
        return
    already_failed = len(failures)
    yield
    new = failures[already_failed:]
    if new:
        pytest.fail(f"{len(new)} failed check(s): {new}")


@pytest.fixture(scope="session")
def db():
    """The UFF4MOF parameter database, loaded once per test session."""
    return load_uff4mof_db()

#!/usr/bin/env python3
"""
Every Python code block of the README and of every documentation page runs
without error, in order, in a fresh directory (docstring examples are
checked separately, by pytest's doctest collection).
"""

import glob
import os

import pytest

from tests.docs_runner import run_page

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = ["README.md"] + sorted(
    os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "docs", "**", "*.md"),
                                                recursive=True))


@pytest.mark.parametrize("page", PAGES)
def test_page_code_runs(page, tmp_path) -> None:
    run_page(os.path.join(ROOT, page), workdir=str(tmp_path))

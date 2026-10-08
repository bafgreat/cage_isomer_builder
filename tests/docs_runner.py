"""Execute the Python code blocks of a Markdown page in order, in one namespace."""
import contextlib
import io
import os
import re
import sys
import tempfile
import warnings

try:                        # plots in the docs must not open windows
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass

BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def blocks(path):
    return BLOCK.findall(open(path, encoding="utf-8").read())


def run_page(path, workdir=None):
    """Run every ```python block of ``path``; return the number run.
    Raises with the failing block's number and code."""
    path = os.path.abspath(path)
    cwd = os.getcwd()
    workdir = workdir or tempfile.mkdtemp()
    os.chdir(workdir)
    namespace = {"__name__": "__docs__"}
    try:
        for k, code in enumerate(blocks(path)):
            try:
                with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exec(compile(code, f"{os.path.basename(path)}[block {k}]", "exec"), namespace)
            except Exception as exc:
                raise RuntimeError(f"{path} block {k} failed: {exc!r}\n---\n{code}") from exc
    finally:
        os.chdir(cwd)
    return len(blocks(path))


if __name__ == "__main__":
    for page in sys.argv[1:]:
        try:
            print("OK  ", page, run_page(page), "blocks")
        except RuntimeError as exc:
            print("FAIL", str(exc).splitlines()[0][:220])

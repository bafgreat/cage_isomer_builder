---
description: >-
  Install cage_isomer_builder from PyPI (pip install cage_isomer_builder)
  or from source on GitHub. Requires Python 3.11+.
---

# Installation

`cage_isomer_builder` requires Python `^3.11` (Python 3.11 or later, but
below Python 4.0). It can be installed directly from PyPI or by cloning the
GitHub repository.

## PyPI

```bash
pip install cage_isomer_builder
```

## GitHub

```bash
git clone https://github.com/bafgreat/cage_isomer_builder.git
cd cage_isomer_builder
pip install .
```

## Optional extras

Two packages are optional, because they have no Windows wheels:

| Extra | Installs | Used for | Without it |
| --- | --- | --- | --- |
| `xtb` | tblite | GFN-xTB optimisation, and the partial charges used by UFF4MOF's Coulomb term | `optimise(method="UFF4MOF")` runs without the Coulomb term, with a warning. `method="GFN2-xTB"` raises an error. |
| `d4` | dftd4 | DFT-D4 scoring in `place_guest_in_host(method="energy")` | Guest sites are scored with the UFF Lennard-Jones term instead, with a warning. |

```bash
pip install "cage_isomer_builder[all]"   # both extras
pip install "cage_isomer_builder[xtb]"   # tblite only
pip install "cage_isomer_builder[d4]"    # dftd4 only
```

## Windows

Install without the extras: `pip install cage_isomer_builder`. Everything
else works, with the fallbacks in the table above.

One required package, pyzeo (used by mofstructure), also has no Windows
wheel. pip then compiles it from source, which needs a C++ compiler such as
the Visual Studio Build Tools. If that fails, use WSL (Windows Subsystem for
Linux), where all packages, including both extras, install normally.

## Building these docs locally

The documentation dependencies are kept in an optional Poetry group so they
don't weigh down a normal install:

```bash
poetry install --with docs
mkdocs serve
```

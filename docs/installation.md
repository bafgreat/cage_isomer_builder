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

## Building these docs locally

The documentation dependencies are kept in an optional Poetry group so they
don't weigh down a normal install:

```bash
poetry install --with docs
mkdocs serve
```

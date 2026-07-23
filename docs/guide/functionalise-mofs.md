# Functionalise MOFs

Existing MOF structures can be loaded directly from a CIF file. The
`functionalise()` method finds the available C-H sites on the linkers and
replaces them with `At` anchor atoms for isomer enumeration. For large unit
cells, use Burnside's lemma to count the symmetry-unique isomers without
generating every structure.

```python
from cage_isomer_builder.cage import CageBuilder

mof = CageBuilder.load_from_file("tests/data/RUBTAK01.cif")

fg_anchors, fg_anchor_indices = mof.functionalise()
print(f"{len(fg_anchor_indices)} functional-group anchor sites detected")

n_unique = mof.count_unique_isomers_burnside()
print(f"{n_unique:.3g} symmetry-unique isomers")

first_few = mof.enumerate_isomers(output_path="mof_isomers", limit=10)
```

The generated files contain the same `At` placeholders used for cage
isomers, so they can be decorated with real fragments using
[`functionalise_isomer_sites()`](functionalising-isomers.md) as shown in the
previous guide page.

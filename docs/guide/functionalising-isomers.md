# Functionalising the enumerated isomers

All `At`-labelled sites in the enumerated isomers can be functionalised
using any functional-group moiety. The only requirement is that the group
must contain an `X` dummy atom at the position to be functionalised.

```python
from ase.io import read
from cage_isomer_builder.utils.functionalise import functionalise_isomer_sites

isomer = read("isomers/0-4-10-12-16-20.xyz")
nh2 = read("tests/data/NH2.xyz")   # a small molecule with one "X" dummy atom

decorated = functionalise_isomer_sites(isomer, fragments=nh2)
```

Several fragment types can be mixed across the active sites by ratio - see
[`functionalise_isomer_sites`](../reference/functionalise.md) for the full
parameter set, including how to track the resulting bond matrix for a
subsequent force-field optimisation of just the new fragments.

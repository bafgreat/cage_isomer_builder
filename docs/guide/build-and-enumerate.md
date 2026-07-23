# Build a cage from building units and enumerate isomers

You can import any of the cage builder classes from
`cage_isomer_builder.cage` to build a cage. The example below builds a
`Tri4Di6` cage.

The isomer enumeration matches all C-H bonds in the linker (courtesy of
[mofstructure](https://github.com/bafgreat/mofstructure)) and enumerates all
unique isomers that can be formed. All unique isomerisation placeholders are
labelled with `At`.

```python
from cage_isomer_builder.cage import Tri4Di6CageBuilder

cage = Tri4Di6CageBuilder(
    node="tests/data/uio66_tri_node.xyz",
    linker="tests/data/bdc.xyz",
)

cage.build()
cage.optimise(rigid_sbus=True)          # UFF4MOF, node/linker shapes preserved
cage.save("cage.xyz")

isomers = cage.enumerate_isomers(output_path="isomers")
print(f"{len(isomers)} symmetry-unique isomers")
```

See the [Topologies](../topologies.md) page for the full list of available
builder classes, and the [API reference](../reference/cage.md) for
`CageBuilder`'s complete method set.

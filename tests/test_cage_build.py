"""
Example script: build a cage, enumerate its isomers, and decorate each
one with real functional groups.

Not a pytest test - it has no test_* functions and asserts nothing, it
just demonstrates the build/enumerate/functionalise pipeline end to end
and writes its output to ./isomers and ./isomers_functionalised. Guarded
under __main__ so pytest importing this file (it matches test_*.py) does
not also run the full build/optimise/enumerate pipeline as a side effect
of collection.

Run directly: python3 tests/test_cage_build.py
"""

import os

from ase import Atoms
from ase.io import read, write

from cage_isomer_builder.cage import Tri4Di6CageBuilder
from cage_isomer_builder.utils.functionalise import functionalise_isomer_sites


def main() -> None:
    cage = Tri4Di6CageBuilder(linker='./tests/data/bdc.xyz', node='./tests/data/uio66_tri_node.xyz')

    cage.build()
    cage.optimise(rigid_sbus=True)
    unique_isomers = cage.enumerate_isomers(output_path='isomers')
    print(f'{len(unique_isomers)} unique isomers written to ./isomers')

    # Decorate each isomer's active ("At") sites with real functional groups -
    # here a 2:1 mix of NH2 and F, chosen randomly per site (seed for
    # reproducibility). Each fragment carries its own "X" atom marking where it
    # bonds to the host.
    nh2 = read('./tests/data/NH2.xyz')
    fluoro = Atoms('XF', positions=[
        [-1.35, 0.00, 0.00],
        [0.00, 0.00, 0.00],
    ])

    os.makedirs('isomers_functionalised', exist_ok=True)
    for filename in os.listdir('isomers'):
        isomer = read(os.path.join('isomers', filename))
        decorated = functionalise_isomer_sites(
            isomer, fragments=[nh2, fluoro], ratios=[2, 0], seed=42,
        )
        write(os.path.join('isomers_functionalised', filename), decorated)

    print(f'{len(unique_isomers)} functionalised isomers written to ./isomers_functionalised')


if __name__ == "__main__":
    main()

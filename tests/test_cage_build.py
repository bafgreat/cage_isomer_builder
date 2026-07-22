import os

from ase import Atoms
from ase.io import read, write

from cage_isomer_builder.cage import Tri4Di6CageBuilder
from cage_isomer_builder.utils.functionalise import functionalise_isomer_sites

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

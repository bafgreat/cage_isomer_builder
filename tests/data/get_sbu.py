from mofstructure.structure import MOFstructure

mof = MOFstructure(filename="RUBTAK01.cif")

sbu = mof.get_sbu(cheminfo=False, add_dummy=False)

sbu[0][0].write("uio66_sbu.xyz")

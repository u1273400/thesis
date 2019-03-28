from vasp import Vasp
from ase.utils.eos import EquationOfState
LC = [3.5, 3.55, 3.6, 3.65, 3.7, 3.75]
energies = []
volumes = []
for a in LC:
    calc = Vasp('bulk/Cu-{0}'.format(a))
    atoms = calc.get_atoms()
    volumes.append(atoms.get_volume())
    energies.append(atoms.get_potential_energy())
calc.stop_if(None in energies)
eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()
print '''
v0 = {0} A^3
E0 = {1} eV
B  = {2} eV/A^3'''.format(v0, e0, B)
eos.plot('images/Cu-fcc-eos.png')
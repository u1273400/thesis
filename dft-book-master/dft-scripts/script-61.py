from vasp import Vasp
from ase import Atom, Atoms
atoms = Atoms([Atom('O', [5, 5, 5])],
              cell=(10, 10, 10))
calc = Vasp('molecules/O',
            xc='PBE',
            encut=400,
            ismear=0,
            atoms=atoms)
E_O = atoms.get_potential_energy()
# now relaxed O2 dimer
atoms = Atoms([Atom('O', [5, 5, 5]),
               Atom('O', [6.22, 5, 5])],
              cell=(10, 10, 10))
calc = Vasp('molecules/O2',
            xc='PBE',
            encut=400,
            ismear=0,
            ibrion=2,
            nsw=10,
            atoms=atoms)
E_O2 = atoms.get_potential_energy()
if None not in (E_O, E_O2):
    print('O2 -> 2O  D = {0:1.3f} eV'.format(2 * E_O - E_O2))
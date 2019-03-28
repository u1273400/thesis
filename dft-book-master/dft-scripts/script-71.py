from vasp import Vasp
from ase import Atom, Atoms
atoms = Atoms([Atom('O', [5.1, 4.2, 6.1], magmom=2)],
              cell=(8, 9, 10))
calc = Vasp('molecules/O-sp-triplet-lowsym',
            xc='PBE',
            encut=400,
            ismear=0,
            sigma=0.01,
            ispin=2,
            atoms=atoms)
E_O = atoms.get_potential_energy()
print('Magnetic moment on O = {0} Bohr  magnetons'.format(atoms.get_magnetic_moment()))
# now relaxed O2 dimer
atoms = Atoms([Atom('O', [5,    5, 5], magmom=1),
               Atom('O', [6.22, 5, 5], magmom=1)],
              cell=(10, 10, 10))
calc = Vasp('molecules/O2-sp-triplet',
            xc='PBE',
            encut=400,
            ismear=0,
            sigma=0.01,
            ispin=2,   # turn spin-polarization on
            ibrion=2,  # make sure we relax the geometry
            nsw=10,
            atoms=atoms)
E_O2 = atoms.get_potential_energy()
# verify magnetic moment
print('Magnetic moment on O2 = {0} Bohr magnetons'.format(atoms.get_magnetic_moment()))
if None not in (E_O, E_O2):
    print('E_O: ', E_O)
    print('O2 -> 2O  D = {0:1.3f} eV'.format(2 * E_O - E_O2))
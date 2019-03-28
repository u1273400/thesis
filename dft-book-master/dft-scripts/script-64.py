from vasp import Vasp
from ase import Atom, Atoms
encuts = [250, 300, 350, 400, 450, 500, 550]
D = []
for encut in encuts:
    atoms = Atoms([Atom('O', [5, 5, 5], magmom=2)],
                  cell=(10, 10, 10))
    calc = Vasp('molecules/O-sp-triplet-{0}'.format(encut),
                xc='PBE',
                encut=encut,
                ismear=0,
                ispin=2,
                atoms=atoms)
    E_O = atoms.get_potential_energy()
    # now relaxed O2 dimer
    atoms = Atoms([Atom('O', [5,    5, 5], magmom=1),
                   Atom('O', [6.22, 5, 5], magmom=1)],
                  cell=(10, 10, 10))
    calc = Vasp('molecules/O2-sp-triplet-{0}'.format(encut),
                xc='PBE',
                encut=encut,
                ismear=0,
                ispin=2,   # turn spin-polarization on
                ibrion=2,  # this turns relaxation on
                nsw=10,
                atoms=atoms)
    E_O2 = atoms.get_potential_energy()
    if None not in (E_O, E_O2):
        d = 2*E_O - E_O2
        D.append(d)
        print('O2 -> 2O encut = {0}  D = {1:1.3f} eV'.format(encut, d))
if not D or None in D: calc.abort()
import matplotlib.pyplot as plt
plt.plot(encuts, D)
plt.xlabel('ENCUT (eV)')
plt.ylabel('O$_2$ dissociation energy (eV)')
plt.savefig('images/O2-dissociation-convergence.png')
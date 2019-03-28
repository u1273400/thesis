# compute ELF for CF4
from vasp import Vasp
from ase.structure import molecule
from enthought.mayavi import mlab
atoms = molecule('CF4')
atoms.center(vacuum=5)
calc = Vasp('molecules/cf4-elf',
            encut=350,
            prec='high',
            ismear=0,
            sigma=0.01,
            xc='PBE',
            lelf=True,
            atoms=atoms)
x, y, z, elf = calc.get_elf()
mlab.contour3d(x, y, z, elf, contours=[0.3])
mlab.savefig('../../images/cf4-elf-3.png')
mlab.figure()
mlab.contour3d(x, y, z, elf, contours=[0.75])
mlab.savefig('../../images/cf4-elf-75.png')
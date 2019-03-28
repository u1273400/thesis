from vasp import Vasp
from ase.lattice.surface import fcc111, add_adsorbate
from ase.constraints import FixAtoms
atoms = fcc111('Pt', size=(2, 2, 3), vacuum=10.0)
# note this function only works when atoms are created by the surface module.
add_adsorbate(atoms, 'O', height=1.2, position='hcp')
constraint = FixAtoms(mask=[atom.symbol != 'O' for atom in atoms])
atoms.set_constraint(constraint)
from ase.io import write
write('images/Pt-hcp-o-site.png', atoms, show_unit_cell=2)
print(Vasp('surfaces/Pt-slab-O-hcp',
           xc='PBE',
           kpts=[4, 4, 1],
           encut=350,
           ibrion=2,
           nsw=25,
           atoms=atoms).potential_energy)
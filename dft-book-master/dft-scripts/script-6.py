from ase.structure import molecule
from ase.io import write
atoms = molecule('CH3CN')
atoms.center(vacuum=6)
print('unit cell')
print('---------')
print(atoms.get_cell())
write('images/ch3cn.png', atoms, show_unit_cell=2)
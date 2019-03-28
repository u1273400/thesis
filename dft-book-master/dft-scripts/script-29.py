from vasp import Vasp
from enthought.mayavi import mlab
from ase.data import vdw_radii
from ase.data.colors import cpk_colors
from ase import Atom, Atoms
atoms = Atoms([Atom('C', [2.422, 0.0, 0.0]),
               Atom('O', [3.578, 0.0, 0.0])],
               cell=(10,10,10))
atoms.center()
calc = Vasp('molecules/co-centered',
            encut=350,
            xc='PBE',
            atoms=atoms)
calc.set(lcharg=True,)
calc.stop_if(calc.potential_energy is None)
atoms = calc.get_atoms()
x, y, z, cd = calc.get_charge_density()
mlab.figure(bgcolor=(1, 1, 1))
# plot the atoms as spheres
for atom in atoms:
    mlab.points3d(atom.x,
                  atom.y,
                  atom.z,
                  scale_factor=vdw_radii[atom.number]/5,
                  resolution=20,
                  # a tuple is required for the color
                  color=tuple(cpk_colors[atom.number]),
                  scale_mode='none')
# draw the unit cell - there are 8 corners, and 12 connections
a1, a2, a3 = atoms.get_cell()
origin = [0, 0, 0]
cell_matrix = [[origin,  a1],
               [origin,  a2],
               [origin,  a3],
               [a1,      a1 + a2],
               [a1,      a1 + a3],
               [a2,      a2 + a1],
               [a2,      a2 + a3],
               [a3,      a1 + a3],
               [a3,      a2 + a3],
               [a1 + a2, a1 + a2 + a3],
               [a2 + a3, a1 + a2 + a3],
               [a1 + a3, a1 + a3 + a2]]
for p1, p2 in cell_matrix:
    mlab.plot3d([p1[0], p2[0]], # x-positions
                [p1[1], p2[1]], # y-positions
                [p1[2], p2[2]], # z-positions
                tube_radius=0.02)
# Now plot the charge density
mlab.contour3d(x, y, z, cd, transparent=True)
# this view was empirically found by iteration
mlab.view(azimuth=-90, elevation=90, distance='auto')
mlab.savefig('images/co-centered-cd.png')
mlab.show()
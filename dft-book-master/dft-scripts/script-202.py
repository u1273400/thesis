from vasp import Vasp
calc = Vasp('surfaces/Pt-slab-O-fcc')
calc.clone('surfaces/Pt-slab-O-fcc-vib')
calc.set(ibrion=5,     # finite differences with selective dynamics
         nfree=2,      # central differences (default)
         potim=0.015,  # default as well
         ediff=1e-8,
         nsw=1)
atoms = calc.get_atoms()
f, v = calc.get_vibrational_modes(0)
print 'Elapsed time = {0} seconds'.format(calc.get_elapsed_time())
allfreq = calc.get_vibrational_modes()[0]
from ase.units import meV
c = 3e10  # cm/s
h = 4.135667516e-15  # eV*s
print 'vibrational energy = {0} eV'.format(f)
print 'vibrational energy = {0} meV'.format(f/meV)
print 'vibrational freq   = {0} 1/s'.format(f/h)
print 'vibrational freq   = {0} cm^{{-1}}'.format(f/(h*c))
print
print 'All energies = ', allfreq
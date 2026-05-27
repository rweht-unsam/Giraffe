# Dinamica Molecular de Agua
# ============================================
import numpy as np
import time
# ============================================
from ase import build
from ase import units
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md import MDLogger
from ase.io.trajectory import Trajectory
from ase.filters  import Filter, UnitCellFilter, ExpCellFilter
from ase.optimize import BFGS, MDMin
# ============================================
from mace.calculators import mace_mp
# ============================================
macemp = mace_mp(model="./MACE-matpes-r2scan-omat-ft.model",default_dtype='float32')
calculator = macemp
# ============================================d
atoms = read('Slab_Water.extxyz')
atoms.pbc=[ 1,1,1 ] 
atoms.calc = macemp
# ============================================
fijos = np.array([], dtype=int)
for atom in atoms:
    if (atom.position[2] > -1.8525 and atom.position[2] < 1.8625):
        fijos = np.append(fijos,int(atom.index))
atoms.set_constraint(FixAtoms(fijos))
# ============================================d
# Initialize velocities.
T_init = 300  # Initial temperature in K
MaxwellBoltzmannDistribution(atoms, T_init * units.kB)
# ============================================d
# Set up the Langevin dynamics engine for NVT ensemble.
dyn = Langevin(atoms, 0.5 * units.fs, T_init * units.kB, 4.0)
# ============================================d
# Define wrap and output
def wrap_atoms(a=atoms):
    a.wrap()
dyn.attach(wrap_atoms,interval=10)
trajectory_file = "Slab_Water.extxyz"
dyn.attach(lambda: write(trajectory_file, atoms, append=True), interval=500)
dyn.attach(MDLogger(dyn, atoms, 'Slab_Water.log', header=False, stress=False,
           peratom=False, mode="a"), interval=500)
# Correr la dinámica
n_steps =  100000 # Number of steps to run
dyn.run(n_steps)
write('CONTCAR_Slab_Water.vasp', atoms)
# Fin dinámica
# ============================================d

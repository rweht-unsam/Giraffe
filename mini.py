# Minimizacion con ASE/MACE
# ============================================
import numpy as np
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# ============================================
from ase import build
from ase import units
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase.filters  import Filter, UnitCellFilter
from ase.optimize import BFGS, MDMin
# ============================================
from mace.calculators import mace_mp
# ============================================
macemp = mace_mp(model="/home/ruweht/mace/mace/calculators/foundations_models/MACE-matpes-r2scan-omat-ft.model",default_dtype='float64')
calculator = macemp
# ============================================

atoms = read('POSCAR', '0')
#atoms.cell=[5.237447,5.237447,10.460767]
atoms.pbc=[ 1,1,1 ] 
atoms.calc = macemp
# ============================================
print("------------------------------------------------")
print("Initial Energy:  ", atoms.get_potential_energy())
print("------------------------------------------------")
ucf = UnitCellFilter(atoms)
optimizer = MDMin(ucf, logfile='mdmin.log')
optimizer.run(fmax=0.001, steps= 50000)# Busqueda de celda de minima energia
print("------------------------------------------------")
print("Optimized Energy (MDMi):", atoms.get_potential_energy())
print("------------------------------------------------")
# save the optimized structure to a file
write('CONTCAR_MDMi.vasp', atoms)

atoms = read('POSCAR', '0')
#atoms.cell=[5.237447,5.237447,10.460767]
atoms.pbc=[ 1,1,1 ] 
atoms.calc = macemp
# ============================================
print("------------------------------------------------")
print("Initial Energy:  ", atoms.get_potential_energy())
print("------------------------------------------------")
ucf = UnitCellFilter(atoms)
optimizer = BFGS(ucf, logfile='mdmin.log')
optimizer.run(fmax=0.001, steps= 50000)
print("------------------------------------------------")
print("Optimized Energy (BFGS):", atoms.get_potential_energy())
print("------------------------------------------------")
# save the optimized structure to a file
write('CONTCAR_BFGS.vasp', atoms)

# 
# Short example showing LS-THC fit of DF coefficients B_pq^A for water in cc-pVDZ/cc-pVDZ-RI
#
import numpy as np
from opt_einsum import contract
from pyscf import gto

import libTHC
import libTHC.grid
import libTHC.ls_thc

# Setup molecule using PySCF
xyz = f'h2o.xyz'
molecule = gto.Mole()
molecule.atom = xyz
molecule.basis = 'cc-pvdz'
molecule.build()

# Generate a small DFT grid from PySCF 
grid_driver = libTHC.grid.Grid(xyz)
grid_driver.type = 'generate'
grid, weights = grid_driver.compute()

# Calculate LS-THC decomposition
ls_thc_driver = libTHC.ls_thc.LS_THC(xyz,grid,weights)
ls_thc_driver.fit_type = '3c'
X, Z = ls_thc_driver.compute()

# Compare LS-THC integrals to conventional integrals
g_conv = molecule.intor('int2e')
g = contract('pa,pb,pq,qc,qd->abcd',X,X,Z,X,X)
print(f'Frobenius loss in (pq|rs): {np.linalg.norm(g_conv-g)}')
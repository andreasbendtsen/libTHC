# 
# Short example showing LS-THC fit of DF coefficients B_pq^A for water in cc-pVDZ/cc-pVDZ-RI
#
import numpy as np
from opt_einsum import contract
from pyscf import gto, df
import scipy 

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
X, Y = ls_thc_driver.compute()

# Compare LS-THC integrals to conventional integrals
g_conv = contract('kp,kq,kA->Apq', X, X, Y)

auxmol = df.addons.make_auxmol(molecule, 'cc-pvdz-ri')
ints_2c2e = auxmol.intor('int2c2e')
ints_3c2e = df.incore.aux_e2(molecule, auxmol, intor='int3c2e')
df_coef = scipy.linalg.solve(
    scipy.linalg.sqrtm(ints_2c2e),
    ints_3c2e.reshape(molecule.nao*molecule.nao,auxmol.nao).T
)
df_coef = df_coef.reshape(auxmol.nao, molecule.nao, molecule.nao)

print(f'Frobenius loss in B_pq^A: {np.linalg.norm(g_conv-df_coef)}')

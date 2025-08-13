import numpy as np
import scipy
from opt_einsum import contract
from pyscf import gto, df

class LS_THC():

    def __init__(self, xyz: str, grid: np.ndarray, weights: np.ndarray):
        """Initialize Least Squares THC driver."""

        # set geometry and grid
        self.xyz = xyz
        self.grid = grid
        self.weights = weights

        # set default settings
        self.fit_type = '4c'
        self.basis = 'cc-pvdz'
        self.auxbasis = 'cc-pvdz-ri'

    def compute(self):

        molecule = gto.Mole()
        molecule.atom = self.xyz
        molecule.basis = self.basis
        molecule.build()

        if self.fit_type == '4c':
            self.setup_X(molecule)
            g = molecule.intor('int2e')
            self.Z = self.ls_thc_4center(g)

        elif self.fit_type == '3c':
            self.setup_X(molecule)
            auxmol = df.addons.make_auxmol(molecule, self.auxbasis)
            ints_2c2e = auxmol.intor('int2c2e')
            ints_3c2e = df.incore.aux_e2(molecule, auxmol, intor='int3c2e')                                                                                               
            df_coef = scipy.linalg.solve(scipy.linalg.sqrtm(ints_2c2e),
                                          ints_3c2e.reshape(molecule.nao*molecule.nao,
                                                             auxmol.nao).T)
            df_coef = df_coef.reshape(auxmol.nao, molecule.nao, molecule.nao)

            self.Z = self.ls_thc_3center(df_coef)         

        elif self.fit_type == '2c':
            self.setup_X(molecule)
            self.setup_X_aux(molecule)
            auxmol = df.addons.make_auxmol(molecule, self.auxbasis)
            ints_2c2e = auxmol.intor('int2c2e')

            self.Z = self.ls_thc_2center(ints_2c2e)

        elif self.fit_type == '2c_chol':
            self.setup_X(molecule)
            self.setup_X_aux(molecule)
            molecule.basis = self.auxbasis
            molecule.build()
            ints_2c2e = molecule.intor('int2c2e')

            L = np.linalg.cholesky(ints_2c2e)
            
            self.Z = self.ls_thc_2center_chol(L)

        elif self.fit_type == '2c_sqrt':
            self.setup_X(molecule)
            auxmol = df.addons.make_auxmol(molecule, self.auxbasis)
            self.setup_X_aux(auxmol)
            ints_2c2e = auxmol.intor('int2c2e')

            L = scipy.linalg.sqrtm(ints_2c2e)

            self.Z = self.ls_thc_2center_chol(L)

        else:
            print('Wrong fit type, so nothing is returned')
            return

        return self.X, self.Z

    def ls_thc_4center(self, ints):
        S = contract('pa,pb,qa,qb->pq',self.X,self.X,self.X,self.X)
        S_inv = scipy.linalg.pinv(S)
        E = contract('pa,pb,abcd,qc,qd->pq',self.X,self.X,ints,self.X,self.X)
        Z = contract('pa,ab,bq->pq',S_inv,E,S_inv)
        return Z

    def ls_thc_3center(self, ints):
        S = contract('pa,pb,qa,qb->pq',self.X,self.X,self.X,self.X)
        E = contract('abc,pb,pc->pa',ints,self.X,self.X)
        xi, *_ = scipy.linalg.lstsq(S,E)
        Z = contract('pa,qa->pq',xi,xi)
        return Z

    def ls_thc_2center(self, ints):
        S = contract('pb,qb->pq',self.Xaux,self.Xaux)
        S_inv = scipy.linalg.pinv(S)
        E = contract('pa,ab,qb->pq',self.Xaux,ints,self.Xaux)
        Z = contract('pa,ab,qb->pq',S_inv,E,S_inv)
        return Z
    
    def ls_thc_2center_chol(self, ints):
        xi, *_ = np.linalg.lstsq(self.Xaux.T, ints, rcond=None)
        Z = contract('pc,qc->pq',xi,xi)
        return Z

    def setup_X(self, molecule):
        self.X = np.sqrt(np.sqrt(self.weights))[:, np.newaxis] * molecule.eval_gto('GTOval_sph', self.grid)
    
    def setup_X_aux(self, auxmol):
        self.Xaux = np.sqrt(self.weights)[:, np.newaxis] * auxmol.eval_gto('GTOval_sph', self.grid)
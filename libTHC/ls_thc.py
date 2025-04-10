import numpy as np
import scipy
from opt_einsum import contract
from pyscf import gto, df

class LS_THC():

    def __init__(self, xyz: np.ndarray, grid: np.ndarray, weights: np.ndarray):
        """Initialize Least Squares THC driver."""
        super().__init__(xyz, grid)

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
        S_inv = scipy.linalg.pinv(S)
        E = contract('abc,pb,pc->pa',ints,self.X,self.X)
        xi = contract('qp,pa->qa',S_inv,E)
        Z = contract('pa,qa->pq',xi,xi)

        return Z

    def ls_thc_2center(self, ints):

        S = contract('pb,qb->pq',self.Xaux,self.Xaux)
        S_inv = scipy.linalg.pinv(S)
        E = contract('pa,ab,qb->pq',self.Xaux,ints,self.Xaux)
        Z = contract('pa,ab,qb->pq',S_inv,E,S_inv)

        return Z

    def setup_X(self, molecule):

        Rao = molecule.eval_gto('GTOval_sph', self.grid)
        Xao = np.zeros_like(Rao)
        for i in range(np.shape(Xao)[1]):
            Xao[:,i] = np.sqrt(np.sqrt(self.weigths)) * Rao[:,i]

        self.X = Xao
    
    def setup_X_aux(self, molecule):

        molecule.basis = self.auxbasis
        molecule.build()
        Raux = molecule.eval_gto('GTOval_sph', self.grid)
        molecule.basis = self.basis
        molecule.build()

        Xaux = np.zeros_like(Raux)
        for i in range(np.shape(Xaux)[1]):
            Xaux[:,i] = np.sqrt(self.weigths) * Raux[:,i]

        self.Xaux = Xaux
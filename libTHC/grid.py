import numpy as np
from opt_einsum import contract
from pyscf import gto, dft
from scipy.optimize import nnls as nnls_scipy
# from grid.becke import BeckeWeights

class Grid():

    def __init__(self, xyz: str):
        """Initialize Least Squares Grid driver."""

        # Set default grid type
        self.type = 'generate'
        self.xyz = xyz
        self.gridfile = None
        self.level = 0

        # Set default reweighting
        self.basis = 'cc-pvdz'
        self.wtol = 1.0e-6
        self.Stol = 1.0e-8

    def compute(self):
        
        if self.type == 'generate':

            self.grid, self.weights = self.generate_grid()

        elif self.type == 'read':
            if self.gridfile == None:
                print('Grid file not set, so nothing is returned')
                return    

            self.grid, self.weights = self.read_grid()
        else:
            print('Wrong grid type, so nothing is returned')
            return    
        
        return self.grid, self.weights

    def read_grid(self):
        
        with open(self.gridfile,'r') as f:
            content = f.readlines()

        ngrid = len(content)
        coord = np.zeros((ngrid,3))
        weights = np.zeros((ngrid))

        for i in range(ngrid):
            line = content[i].strip().split()
            coord[i,0] = float(line[0])
            coord[i,1] = float(line[1])
            coord[i,2] = float(line[2])
            weights[i] = float(line[3])

        return coord, weights
    
    def generate_grid(self):

        # Setup molecule
        molecule = gto.Mole()
        molecule.atom = self.xyz
        molecule.basis = self.basis
        molecule.build()

        # Setup dft driver
        mf = dft.RKS(molecule)
        mf.verbose = 0
        mf.grids.level = self.level
        mf.kernel()

        coord = mf.grids.coords
        weights = mf.grids.weights
        coord = coord[weights>0,:]
        weights = weights[weights>0]        

        return coord, weights
    
    def reweight_grid(self, type: str, scheme: str, use_gpu: bool = True, resname = "nnls_checkpoint.npz"):

        if type == 'overlap':

            # Setup molecule
            molecule = gto.Mole()
            molecule.atom = self.xyz
            molecule.basis = self.basis
            molecule.build()

            # Determine overlap mat and overlap on a grid
            S = molecule.intor('int1e_ovlp')
            R = molecule.eval_gto('GTOval_sph', self.grid)

            S_grid = contract('xp,xq,x->pq',R,R,self.weights)

            print('Standard S error primary basis: ',np.linalg.norm(S-S_grid))

            nao = np.shape(R)[1]
            ngrid = np.shape(R)[0]
            R_concat = contract('xp,xq->xpq',R,R).reshape(ngrid,nao*nao)

            A = R_concat.T
            b = S.reshape(nao*nao)

            if scheme == 'nnls':
                if use_gpu:
                    x = nnls_fast_gpu(A,b,tol=self.wtol,Stol=self.Stol,checkpoint_file=resname)
                else:
                    x = nnls_fast(A,b,tol=self.wtol,Stol=self.Stol,checkpoint_file=resname)
            elif scheme == 'nnls_scipy':
                x, res = nnls_scipy(A,b,10*ngrid)
            else:
                print('Wrong reweighting scheme, so nothing is returned')
                return

            S_grid_new_x = contract('xp,xq,x->pq',R,R,x)

            print('Optimized S error primary basis: ',np.linalg.norm(S-S_grid_new_x)) 
            print(f'Discarding {np.shape(x)[0]-np.shape(x[x!=0])[0]} points of {np.shape(x)[0]} points \n')

            self.grid = self.grid[x!=0,:]
            self.weights = x[x!=0]

        else:
            print('Wrong reweighting type, so nothing is returned')
            return

        return self.grid, self.weights
    
import numpy as np
from scipy.linalg import cho_factor, cho_solve, lstsq
import sys
import os

def save_checkpoint(x, passive, i, filename="nnls_checkpoint.npz"):
    np.savez(filename,
             x=x,
             passive=passive,
             i=np.array([i]))

def load_checkpoint(filename="nnls_checkpoint.npz"):
    data = np.load(filename)
    x = np.asarray(data["x"])
    passive = np.asarray(data["passive"])
    i = int(data["i"][0])
    return x, passive, i

def nnls_fast(A, b, tol=1e-6, Stol=1e-8, max_iter=None, checkpoint_file="nnls_checkpoint.npz"):
    A = np.asfortranarray(A)
    b = np.asfortranarray(b)
    m, n = A.shape

    # Try to load checkpoint
    print(checkpoint_file)
    if os.path.exists(checkpoint_file):
        print("Loading from checkpoint...")
        x, passive, start_iter = load_checkpoint(checkpoint_file)
        r = b - A @ x
        w = A.T @ r
    else:
        x = np.zeros(n)
        passive = np.zeros(n, dtype=bool)
        r = b.copy()
        w = A.T @ r
        start_iter = 0

    if max_iter is None:
        max_iter = 5*n

    for _ in range(start_iter,max_iter):
        if _ % 1 == 0:
            npas = np.sum((~passive & (w <= tol)) | passive)
            ntot = np.shape(x)[0]
            print(f'Iter: {_} of {max_iter} with {npas} out of {ntot} weights converged')
            sys.stdout.flush()
        # Save every 100 iterations
        if _ % 10 == 0:
            save_checkpoint(x, passive, _, checkpoint_file)
        # Optimality check: w <= tol for all inactive
        if np.all((~passive & (w <= tol)) | passive):
            print('Converging as all weights are passive')
            break

        # Activate index with largest gradient violation
        t = np.argmax(np.where(~passive, w, -np.inf))
        passive[t] = True

        while True:
            P = np.flatnonzero(passive)
            Ap = A[:, P]

            # Solve LS via Cholesky
            G = Ap.T @ Ap
            c = Ap.T @ b
            try:
                c_factor = cho_factor(G)
                z_p = cho_solve(c_factor, c)
            except np.linalg.LinAlgError:
                z_p, *_ = lstsq(Ap, b)

            if np.all(z_p >= 0):
                x[:] = 0
                x[P] = z_p
                break

            z = np.zeros_like(x)
            z[P] = z_p

            # Step to boundary
            mask = (z < 0) & passive
            step = np.min(x[mask] / (x[mask] - z[mask] + 1e-12))
            x += step * (z - x)
            passive[x < tol] = False

        r = b - A @ x
        w = A.T @ r

    return x

import os
import cupy as cp

def save_checkpoint_gpu(x, passive, i, filename="nnls_checkpoint.npz"):
    np.savez(filename,
             x=cp.asnumpy(x),
             passive=cp.asnumpy(passive),
             i=np.array([i]))

def load_checkpoint_gpu(filename="nnls_checkpoint.npz"):
    data = np.load(filename)
    x = cp.asarray(data["x"])
    passive = cp.asarray(data["passive"])
    i = int(data["i"][0])
    return x, passive, i

def nnls_fast_gpu(A, b, tol=1e-6, Stol=1e-8, max_iter=None, checkpoint_file="nnls_checkpoint.npz"):
    A = cp.asarray(A)
    b = cp.asfortranarray(cp.asarray(b))
    m, n = A.shape
    npas = 0

    # Try to load checkpoint
    if os.path.exists(checkpoint_file):
        print("Loading from checkpoint...")
        x, passive, start_iter = load_checkpoint_gpu(checkpoint_file)
        r = b - A @ x
        w = A.T @ r
    else:
        x = cp.zeros(n)
        passive = cp.zeros(n, dtype=bool)
        r = b.copy()
        w = A.T @ r
        start_iter = 0

    if max_iter is None:
        max_iter = 5*n

    for _ in range(start_iter,max_iter):
        if _ % 10 == 0:
            npas = int(cp.sum((~passive & (w <= tol)) | passive).get())
            ntot = x.shape[0]
            print(f'Iter: {_} of {max_iter} with {npas} out of {ntot} weights converged')
            sys.stdout.flush()

        # Save every 100 iterations
        if _ % 100 == 0:
            save_checkpoint_gpu(x, passive, _, checkpoint_file)

        if cp.all((~passive & (w <= tol)) | passive):
            print('Converging as all weights are passive')
            break

        t = int(cp.argmax(cp.where(~passive, w, -cp.inf)))
        passive[t] = True

        while True:
            P = cp.flatnonzero(passive)
            Ap = cp.asfortranarray(A[:, P])
            
            try:
                z_p = cp.linalg.lstsq(Ap, b, rcond=None)[0]
            except Exception as e:
                Q, R = cp.linalg.qr(Ap)
                z_p = cp.linalg.solve(R, Q.T @ b)

            if cp.all(z_p >= 0):
                x[:] = 0
                x[P] = z_p
                break

            z = cp.zeros_like(x)
            z[P] = z_p

            mask = (z < 0) & passive
            step = cp.min(x[mask] / (x[mask] - z[mask] + 1e-12))
            x += step * (z - x)
            passive[x < tol] = False

        r = b - A @ x
        w = A.T @ r

    return cp.asnumpy(x)


# def get_IP_QR(molecule,coords,weights,epsilon):

#     Rao = molecule.eval_gto('GTOval_sph', coords)

#     rho = contract('xi,xj->xij',Rao,Rao)
#     rho = contract('xij,x->xij',rho,weights).reshape(np.shape(weights)[0],molecule.nao*molecule.nao)

#     Q, R, E = scipy.linalg.qr(rho.T, pivoting=True)

#     for i in range(molecule.nao**2-1):
#         if np.linalg.norm(R[i+1,i+1]) < epsilon:
#             Naux = i+1
#             break

#     IP = np.zeros((Naux,3))
#     IP_weights = np.zeros((Naux,))
#     for i in range(Naux):
#         index = E[i]
#         IP[i,:] = coords[index,:]
#         IP_weights[i] = weights[index]

#     print(Naux)

#     return IP, IP_weights


# def get_IP_CVT_v2(molecule,cIP):

#     aux_list = [14,56]
#     nIP_arr = np.zeros((molecule.natm))
#     for atom in range(molecule.natm):
#         if molecule.atom_charge(atom) == 8:
#             nIP_arr[atom] = int(aux_list[1]*cIP)
#         else:
#             nIP_arr[atom] = int(aux_list[0]*cIP)

#     nIP = int(np.sum(nIP_arr))

#     IP_full = np.zeros((nIP,3))
#     IP_weights_full = np.ones((nIP))
#     p_index = [0]

#     index1 = 0
#     index2 = int(nIP_arr[0])

#     for atom in range(molecule.natm):

#         # set some atomic info
#         atom_coords = molecule.atom_coord(atom)
#         atom_charge = molecule.atom_charge(atom)

#         mol_temp = gto.Mole()
#         mol_temp.atom = f'''{atom_charge} {atom_coords[0]} {atom_coords[1]} {atom_coords[2]}'''
#         if atom_charge==1:
#             mol_temp.charge = -1
#         mol_temp.build()

#         grids = dft.gen_grid.Grids(mol_temp)
#         grids.level = 1
#         grids.build()

#         coords = grids.coords
#         weights = grids.weights

#         Rao = molecule.eval_gto('GTOval_sph', coords)
#         weights = weights*np.sum(Rao*Rao,axis=1)
        
#         # Choose nIP random initial centroids
#         number_of_rows = coords.shape[0] 
#         random_indices = np.random.choice(number_of_rows,size=int(nIP_arr[atom]),replace=False) 
        
#         IP = np.zeros((int(nIP_arr[atom]),3))
#         for i in range(int(nIP_arr[atom])):
#             index = random_indices[i]
#             IP[i,:] = coords[index,:]  

#         # Loop to converge controids
#         iter = 0
#         Fswitch = 1.0
#         cindex = np.zeros(np.shape(coords)[0])
#         while (iter < 1000 and Fswitch > 0.001):

#             IP_old = IP.copy()
#             cindex_old = cindex.copy()

#             Nswitch = 0
#             for i in range(np.shape(coords)[0]):
#                 r = coords[i,:]
#                 cindex[i] = np.argmin(np.linalg.norm(r-IP,axis=1))
#                 if cindex[i] != cindex_old[i]:
#                     Nswitch += 1

#             Fswitch = Nswitch/np.shape(coords)[0]

#             for i in range(int(nIP_arr[atom])):
#                 IP[i,:] = 0.0
#                 N = 0.0
#                 for j in range(np.shape(coords)[0]):
#                     if cindex[j] == i:
#                         IP[i,:] += coords[j,:]*weights[j]
#                         N += weights[j]

#                 if N == 0:
#                     IP[i,:] = IP_old[i,:]
#                 else:
#                     IP[i,:] = IP[i,:]/N

#             iter += 1

#             print(np.linalg.norm(IP-IP_old), Fswitch)

#         IP_full[index1:index2,:] = IP
#         p_index.append(index2)

#         index1 += int(nIP_arr[atom])
#         if atom != molecule.natm-1:
#             index2 += int(nIP_arr[atom+1])

#     IP_weights_full = BeckeWeights().compute_weights(IP_full, molecule.atom_coords(), molecule.atom_charges(), pt_ind=p_index)
#     # IP_weights_full = HirshfeldWeights().generate_proatom(IP, atom_coords, atom_charge)
    
#     print(nIP)

#     return IP_full, IP_weights_full
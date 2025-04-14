import numpy as np
from opt_einsum import contract
from pyscf import gto, dft
from scipy.optimize import nnls, lsq_linear
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
# from grid.becke import BeckeWeights

class Grid():

    def __init__(self):
        """Initialize Least Squares Grid driver."""

        # Set default grid type
        self.type = 'read'
        self.gridfile = None

        # Set default reweighting
        self.basis = 'cc-pvdz'

    def compute(self):

        if self.type == 'read':
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
    
    def reweight_grid(self, xyz: str, type: str):

        if type == 'overlap':

            # Setup molecule
            molecule = gto.Mole()
            molecule.atom = xyz
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

            # x, _ = nnls(A, b, 10000)

            # result = lsq_linear(A, b, bounds=(0, np.inf), method='bvls')  # or 'bvls'
            # x = result.x

            scaler = StandardScaler()
            A_scaled = scaler.fit_transform(A)
            b_mean = b.mean()
            b_scaled = b - b_mean

            model = LinearRegression(positive=True)
            model.fit(A_scaled, b_scaled)

            # Get scaled coefficients
            beta_scaled = model.coef_

            # Unscale the coefficients
            x = beta_scaled / scaler.scale_

            S_grid_new_x = contract('xp,xq,x->pq',R,R,x)

            print('Optimized S error primary basis: ',np.linalg.norm(S-S_grid_new_x)) 
            print(f'Discarding {np.shape(x)[0]-np.shape(x[x>0])[0]} points of {np.shape(x)[0]} points \n')

            self.grid = self.grid[x>0,:]
            self.weights = x[x>0]

        else:
            print('Wrong reweighting type, so nothing is returned')
            return

        return self.grid, self.weights


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
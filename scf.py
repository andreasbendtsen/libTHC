import numpy as np
from pyscf import scf
import scipy
from opt_einsum import contract

def c1diis(F_mats, e_vecs, norb):

    n = len(e_vecs)

    # build DIIS matrix
    B = -np.ones((n + 1, n + 1))
    B[n, n] = 0

    for i in range(n):
        for j in range(n):
            B[i, j] = np.dot(e_vecs[i], e_vecs[j])

    b = np.zeros(n + 1)
    b[n] = -1

    w = np.matmul(np.linalg.inv(B), b)

    F_diis = np.zeros((norb, norb))
    for i in range(n):
        F_diis += w[i] * F_mats[i]

    return F_diis

def run_scf(molecule,g,max_iter=50,conv_thresh=1e-6,init_guess='hcore'):

    # setup
    norb = molecule.nao
    nocc = int(molecule.nelectron/2)
    V_nuc = molecule.enuc

    print("Number of contracted basis functions:", norb)
    print("Number of doubly occupied molecular orbitals:", nocc)
    print(f"Nuclear repulsion energy (in a.u.): {V_nuc : 14.12f}")

    # overlap matrix
    S = molecule.intor('int1e_ovlp')

    # one-electron Hamiltonian
    T = molecule.intor('int1e_kin')
    V = molecule.intor('int1e_nuc')
    h = T + V

    e_vecs = []
    F_mats = []
    error_DIIS = []

    # initial guess from core Hamiltonian
    if init_guess == 'hcore':
        epsilon, C = scipy.linalg.eigh(h, S)
        D = np.einsum("ik,jk->ij", C[:, :nocc], C[:, :nocc])
    elif init_guess == 'sad':
        D = scf.get_init_guess(molecule,'minao',s1e=S)
    elif init_guess == 'atom':
        D = scf.get_init_guess(molecule,'atom',s1e=S)
    elif init_guess == 'huckel':
        D = scf.get_init_guess(molecule,'huckel',s1e=S)
    elif init_guess == 'mod_huckel':
        D = scf.get_init_guess(molecule,'mod_huckel',s1e=S)
    elif init_guess == 'sap':
        D = scf.get_init_guess(molecule,'sap',s1e=S)
    

    print("iter      SCF energy    Error norm")



    for iter in range(max_iter):

        J = np.einsum("ijkl,kl->ij", g, D)
        K = np.einsum("ilkj,kl->ij", g, D)
        F = h + 2 * J - K
        F_mats.append(F)

        E = np.einsum("ij,ij->", h + F, D) + V_nuc

        # compute convergence metric
        e_mat = np.linalg.multi_dot([F, D, S]) - np.linalg.multi_dot([S, D, F])
        e_vecs.append(e_mat.reshape(-1))
        error = np.linalg.norm(e_vecs[-1])

        print(f"{iter:>2d}  {E:16.12f}  {error:10.2e}")
        error_DIIS.append(error)

        if error < conv_thresh:
            print("SCF iterations converged!")
            break

        F = c1diis(F_mats, e_vecs, norb)
        epsilon, C = scipy.linalg.eigh(F, S)

        D = np.einsum("ik,jk->ij", C[:, :nocc], C[:, :nocc])

    return E


def run_dscf(molecule,g_full,g,max_iter=50,conv_thresh=1e-6,init_guess='sad'):

    # setup
    norb = molecule.nao
    nocc = int(molecule.nelectron/2)
    V_nuc = molecule.enuc

    print("Number of contracted basis functions:", norb)
    print("Number of doubly occupied molecular orbitals:", nocc)
    print(f"Nuclear repulsion energy (in a.u.): {V_nuc : 14.12f}")

    # overlap matrix
    S = molecule.intor('int1e_ovlp')

    # one-electron Hamiltonian
    T = molecule.intor('int1e_kin')
    V = molecule.intor('int1e_nuc')
    h = T + V

    e_vecs = []
    F_mats = []
    error_DIIS = []

    # Density of external system 
    if init_guess == 'hcore':
        epsilon, C = scipy.linalg.eigh(h, S)
        D0 = np.einsum("ik,jk->ij", C[:, :nocc], C[:, :nocc])
    elif init_guess == 'sad':
        D0 = scf.get_init_guess(molecule,'minao',s1e=S)
    elif init_guess == 'atom':
        D0 = scf.get_init_guess(molecule,'atom',s1e=S)
    elif init_guess == 'huckel':
        D0 = scf.get_init_guess(molecule,'huckel',s1e=S)
    elif init_guess == 'mod_huckel':
        D0 = scf.get_init_guess(molecule,'mod_huckel',s1e=S)
    elif init_guess == 'sap':
        D0 = scf.get_init_guess(molecule,'sap',s1e=S)

    # External potential V0
    V0 = V + 2*contract('pqrs,rs->pq',g_full,D0) - contract('prqs,rs->pq',g_full,D0)

    # External Core hamiltonian H0
    H0 = T + V0

    # External energy
    E0 = V_nuc + 2 * contract('pq,pq->',D0,T+V) + 2 * contract('pq,pqrs,rs->',D0,g_full,D0) - contract('pq,prqs,rs->',D0,g_full,D0)

    # Iterate to obtain difference density deltaD
    deltaD = np.zeros_like(D0)
    dE = 0.0
    dE_old = 0.0

    print("iter      SCF energy      dSCF energy    Error norm")

    for iter in range(max_iter):

        J = np.einsum("pqrs,rs->pq", g, deltaD)
        K = np.einsum("prqs,rs->pq", g, deltaD)
        F = H0 + 2*J - K
        F_mats.append(F)

        dE = np.einsum("pq,pq->", H0 + F, deltaD)

        # compute convergence metric
        e_mat = np.linalg.multi_dot([F, deltaD+D0, S]) - np.linalg.multi_dot([S, deltaD+D0, F])
        e_vecs.append(e_mat.reshape(-1))
        error = np.linalg.norm(e_vecs[-1])

        E = E0 + dE

        print(f"{iter:>2d}  {E:16.12f}  {dE:16.12f}  {error:10.2e}")
        error_DIIS.append(error)

        if iter > 1:
            if error < conv_thresh:
                print("SCF iterations converged!")
                break

        F = c1diis(F_mats, e_vecs, norb)
        epsilon, C = scipy.linalg.eigh(F, S)

        deltaD = np.einsum("ik,jk->ij", C[:, :nocc], C[:, :nocc]) - D0

    return E
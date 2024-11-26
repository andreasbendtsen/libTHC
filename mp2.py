from opt_einsum import contract

def run_mp2(molecule,g,epsilon,C):

    # setup
    norb = molecule.nao
    nocc = int(molecule.nelectron/2)
    nvirt = norb - nocc
    Emp2 = 0.0
    e_mp2_os = 0.0
    e_mp2_ss = 0.0
    co = C[:,:nocc]
    cv = C[:,nocc:]
    eocc = epsilon[:nocc]
    evirt = epsilon[nocc:]

    ovov = contract('pqrs,pi,qa,rj,sb->iajb',g,co,cv,co,cv)  

    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvirt):
                for b in range(nvirt):
                    e_ijab = evirt[a] + evirt[b] - eocc[i] - eocc[j]

                    e_mp2_os -= (ovov[i, a, j, b] * ovov[i, a, j, b]) / e_ijab

                    e_mp2_ss -= (
                        ovov[i, a, j, b] * (ovov[i, a, j, b] - ovov[i, b, j, a]) / e_ijab
                    )

    Emp2 = e_mp2_os + e_mp2_ss

    return Emp2
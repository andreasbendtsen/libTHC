import numpy as np
import scipy
from opt_einsum import contract

def ls_thc_4center(ints, X):
    
    S = contract('pa,pb,qa,qb->pq',X,X,X,X)
    S_inv = scipy.linalg.pinv(S)
    E = contract('pa,pb,abcd,qc,qd->pq',X,X,ints,X,X)
    Z = contract('pa,ab,bq->pq',S_inv,E,S_inv)

    return Z

def ls_thc_3center(ints, X):
    
    S = contract('pa,pb,qa,qb->pq',X,X,X,X)
    S_inv = scipy.linalg.pinv(S)
    E = contract('abc,pb,pc->pa',ints,X,X)
    xi = contract('qp,pa->qa',S_inv,E)
    Z = contract('pa,qa->pq',xi,xi)

    return Z

def ls_thc_2center(ints, X):

    S = contract('pb,qb->pq',X,X)
    S_inv = scipy.linalg.pinv(S)
    E = contract('pa,ab,qb->pq',X,ints,X)
    Z = contract('pa,ab,qb->pq',S_inv,E,S_inv)

    return Z
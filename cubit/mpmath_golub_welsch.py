from mpmath import mp
from mpmath.matrices.eigen_symmetric import tridiag_eigen

def gauss_laguerre_nodes(n):
    d = mp.arange(1,2*n,2)
    e = mp.arange(-1,-n-1,-1)
    tridiag_eigen(mp, d, e)
    return d

def gauss_genlaguerre(n, alpha):
    d = [mp.mpf(i) + alpha for i in range(1,2*n,2)]
    e = [-mp.sqrt(mp.mpf(k)*(k + alpha)) for k in range(1,n)]
    e.append(mp.mpf('0.0'))
    tridiag_eigen(mp, d, e)
    return d

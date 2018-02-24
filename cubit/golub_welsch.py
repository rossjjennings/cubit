from mpmath import mp
from mpmath.matrices.eigen_symmetric import tridiag_eigen

def gauss_hermite(n):
    d = mp.matrix([mp.mpf('0.0') for _ in range(n)])
    e = [mp.sqrt(k/2) for k in mp.arange(1,n)]
    e.append(mp.mpf('0.0'))
    e = mp.matrix(e)
    z = mp.eye(n)[0,:]
    
    tridiag_eigen(mp, d, e, z)
    z = mp.sqrt(mp.pi)*z.apply(lambda x: x**2)
    return d, z.T

def gauss_laguerre(n):
    d = mp.matrix(mp.arange(1,2*n,2))
    e = mp.matrix(mp.arange(-1,-n-1,-1))
    z = mp.eye(n)[0,:]
    
    tridiag_eigen(mp, d, e, z)
    z = z.apply(lambda x: x**2)
    return d, z.T

def gauss_genlaguerre(n, alpha):
    d = mp.matrix([i + alpha for i in mp.arange(1,2*n,2)])
    e = [-mp.sqrt(k*(k + alpha)) for k in mp.arange(1,n)]
    e.append(mp.mpf('0.0'))
    e = mp.matrix(e)
    z = mp.eye(n)[0,:]
    
    tridiag_eigen(mp, d, e, z)
    z = mp.gamma(alpha + 1)*z.apply(lambda x: x**2)
    return d, z.T

def gauss_legendre(n):
    d = mp.matrix([mp.mpf('0.0') for _ in range(n)])
    e = [k/mp.sqrt(4*k**2 - 1) for k in mp.arange(1,n)]
    e.append(mp.mpf('0.0'))
    e = mp.matrix(e)
    z = mp.eye(n)[0,:]
    
    tridiag_eigen(mp, d, e, z)
    z = 2*z.apply(lambda x: x**2)
    return d, z.T

def gauss_gegenbauer(n, alpha):
    d = mp.matrix([mp.mpf('0.0') for _ in range(n)])
    e = [mp.sqrt(k*(k + 2*alpha - 1)/((2*k + 2*alpha - 1)**2 - 1))
         for k in mp.arange(1,n)]
    e.append(mp.mpf('0.0'))
    e = mp.matrix(e)
    z = mp.eye(n)[0,:]
    
    tridiag_eigen(mp, d, e, z)
    z = 2**(2*alpha)*mp.beta(alpha + 1/2, alpha + 1/2)*z.apply(lambda x: x**2)
    return d, z.T

def gauss_jacobi(n, alpha, beta):
    d = mp.matrix([(beta**2 - alpha**2)/(2*k + alpha + beta)/(2*k + alpha + beta - 2)
                   for k in mp.arange(1,n+1)])
    e = [k*(k + alpha)*(k + beta)*(k + alpha + beta)/((2*k + alpha + beta)**2 - 1)
         for k in mp.arange(1,n)]
    e = [2/(2*k + alpha + beta)*mp.sqrt(x) for k, x in zip(mp.arange(1,n), e)]
    e.append(mp.mpf('0.0'))
    e = mp.matrix(e)
    z = mp.eye(n)[0,:]
    
    tridiag_eigen(mp, d, e, z)
    z = 2**(alpha + beta + 1)*mp.beta(alpha + 1, beta + 1)*z.apply(lambda x: x**2)
    return d, z.T

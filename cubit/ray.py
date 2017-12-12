import numpy as np
from numpy.polynomial import laguerre
from numpy import pi, sin, cos, exp, log, sqrt
from scipy import linalg, special

def gauss_laguerre(n):
    '''
    Gauss-Laguerre quadrature:
    
    A rule of order 2*n-1 on the ray with respect to
    the weight function w(x) = exp(-x).
    '''
    return laguerre.laggauss(n)

def gauss_genlaguerre(n, alpha):
    '''
    Generalized Gauss-Laguerre quadrature:
    
    A rule of order 2*n-1 on the ray with respect to
    the weight function w(x) = x**alpha*exp(-x).
    '''
    k = np.arange(1, n+1)
    J_bands = np.zeros((2, n))
    J_bands[0,1:] = -sqrt(k*(k + alpha))[:-1]
    J_bands[1,:] = 2*k + alpha - 1
    nodes, vectors = linalg.eig_banded(J_bands)
    weights = special.gamma(alpha + 1)*vectors[0,:]**2
    
    return nodes, weights

def exponential(n, scale=1):
    '''
    Gauss-Laguerre quadrature:
    
    A rule of order 2*n-1 on the ray with respect to the PDF of an
    exponential distribution with arbitrary scale.
    '''
    nodes, weights = laguerre.laggauss(n)
    nodes = scale*nodes
    
    return nodes, weights

def gamma(n, alpha, scale=1):
    '''
    Generalized Gauss-Laguerre quadrature:
    
    A rule of order 2*n-1 on the ray with respect to the PDF of a
    gamma distribution with arbitrary scale and shape parameter `alpha`.
    '''
    k = np.arange(1, n+1)
    J_bands = np.zeros((2, n))
    J_bands[0,1:] = -sqrt(k*(k + alpha))[:-1]
    J_bands[1,:] = 2*k + alpha - 2
    nodes, vectors = linalg.eig_banded(J_bands)
    weights = vectors[0,:]**2
    nodes *= scale
    
    return nodes, weights

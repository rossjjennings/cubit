import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt
from scipy import linalg, special

def gauss_laguerre(n):
    '''
    Gauss-Laguerre quadrature:
    
    A rule of order 2*n-1 on the ray with respect to
    the weight function w(x) = exp(-x).
    '''
    return special.roots_laguerre(n)

def gauss_genlaguerre(n, alpha):
    '''
    Generalized Gauss-Laguerre quadrature:
    
    A rule of order 2*n-1 on the ray with respect to
    the weight function w(x) = x**alpha*exp(-x).
    '''
    return special.roots_genlaguerre(n, alpha)

def exponential(n, scale=1):
    '''
    Gauss-Laguerre quadrature:
    
    A rule of order 2*n-1 on the ray with respect to the PDF of an
    exponential distribution with arbitrary scale.
    '''
    nodes, weights = special.roots_laguerre(n)
    nodes = scale*nodes
    
    return nodes, weights

def gamma(n, alpha, scale=1):
    '''
    Generalized Gauss-Laguerre quadrature:
    
    A rule of order 2*n-1 on the ray with respect to the PDF of a
    gamma distribution with arbitrary scale and shape parameter `alpha`.
    '''
    return special.roots_genlaguerre(n, alpha)
    nodes *= scale
    weights /= np.sum(weights)
    
    return nodes, weights

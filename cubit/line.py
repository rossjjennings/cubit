import numpy as np
from scipy import special

def gauss_hermite(n):
    '''
    Gauss-Hermite quadrature:
    
    A rule of order 2*n-1 on the line with respect to
    the weight function w(x) = exp(-x**2).
    '''
    return special.roots_hermite(n)

def gauss_hermite_e(n):
    '''
    Gauss-Hermite quadrature:
    
    A rule of order 2*n-1 on the line with respect to
    the weight function w(x) = exp(-x**2/2).
    '''
    return special.roots_hermitenorm(n)

def normal(n, loc=0, scale=1):
    '''
    Gauss-Hermite quadrature:
    
    A rule of order 2*n-1 on the line with respect to the PDF of a
    normal distribution with arbitrary location and scale.
    '''
    nodes, weights = special.roots_hermitenorm(n)
    nodes = loc + scale*nodes
    weights = weights/np.sqrt(2*np.pi)
    return nodes, weights

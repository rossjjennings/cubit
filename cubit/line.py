import numpy as np
from numpy.polynomial import hermite, hermite_e

def hermgauss(n):
    '''
    Gauss-Hermite quadrature:
    
    A rule of order 2*n + 1 on the line with respect to
    the weight function w(x) = exp(-x**2).
    '''
    return hermite.hermgauss(n)

def hermegauss(n):
    '''
    Gauss-Hermite quadrature:
    
    A rule of order 2*n + 1 on the line with respect to
    the weight function w(x) = exp(-x**2/2).
    '''
    return hermite_e.hermegauss(n)

def gaussian(n, loc=0, scale=1):
    '''
    Gauss-Hermite quadrature:
    
    A rule of order 2*n + 1 on the line with respect to a Gaussian
    weight function with arbitrary location and scale. Normalized
    so that the sum of the weights is 1.
    '''
    nodes, weights = hermite_e.hermegauss(n)
    nodes = loc + scale*nodes
    weights = weights/np.sqrt(2*np.pi)
    return nodes, weights

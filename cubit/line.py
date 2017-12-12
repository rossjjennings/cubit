import numpy as np
from numpy.polynomial import hermite, hermite_e

def gauss_hermite(n):
    '''
    Gauss-Hermite quadrature:
    
    A rule of order 2*n-1 on the line with respect to
    the weight function w(x) = exp(-x**2).
    '''
    return hermite.hermgauss(n)

def gauss_hermite_e(n):
    '''
    Gauss-Hermite quadrature:
    
    A rule of order 2*n-1 on the line with respect to
    the weight function w(x) = exp(-x**2/2).
    '''
    return hermite_e.hermegauss(n)

def normal(n, loc=0, scale=1):
    '''
    Gauss-Hermite quadrature:
    
    A rule of order 2*n-1 on the line with respect to the PDF of a
    normal distribution with arbitrary location and scale.
    '''
    nodes, weights = hermite_e.hermegauss(n)
    nodes = loc + scale*nodes
    weights = weights/np.sqrt(2*np.pi)
    return nodes, weights

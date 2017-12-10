import numpy as np
from numpy.polynomial import laguerre

def laggauss(n):
    '''
    Gauss-Laguerre quadrature:
    
    A rule of order 2*n + 1 on the ray with respect to
    the weight function w(x) = exp(-x).
    '''
    return laguerre.laggauss(n)

def exponential(n, scale=1):
    '''
    Gauss-Laguerre quadrature:
    
    A rule of order 2*n + 1 on the ray with respect to an exponential
    weight function with arbitrary scale.
    '''
    nodes, weights = laguerre.laggauss(n)
    nodes = scale*nodes
    return nodes, weights
    
    

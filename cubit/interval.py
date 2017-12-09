import numpy as np
from numpy.polynomial import legendre, chebyshev

def gauss(n, lower=-1, upper=1):
    '''
    Gaussian quadrature. This function aliases `leggauss`.
    '''
    return leggauss(n, lower, upper)

def leggauss(n, lower=-1, upper=1):
    '''
    Gauss-Legendre quadrature:
    
    A rule of order 2*n-1 on the interval [lower, upper] 
    with respect to the weight function w(x) = 1.
    '''
    nodes, weights = legendre.leggauss(n)
    if lower != -1 or upper != 1:
        nodes = (upper+lower)/2 + (upper-lower)/2*nodes
        weights = (upper-lower)/2*weights
    return nodes, weights

def chebgauss(n, lower=-1, upper=1):
    '''
    Gauss-Chebyshev quadrature:
    
    A rule of order 2*n-1 on the interval [lower, upper]
    with respect to the weight function w(x) = 1/sqrt(1-x**2).
    '''
    nodes, weights = chebyshev.chebgauss(n)
    if lower != -1 or upper != 1:
        nodes = (upper+lower)/2 + (upper-lower)/2*nodes
        weights = (upper-lower)/2*weights
    return nodes, weights

def trapz(m, lower=-1, upper=1):
    '''
    Trapezoid rule:
    
    A well-known composite rule on the interval [lower, upper],
    this is the first of the Newton-Cotes rules.
    The total number of evaluation points is m + 1.
    '''
    h = (upper-lower)/m
    nodes = np.linspace(lower, upper, m+1)
    weights = np.full(m+1, h)
    weights[0] = weights[-1] = h/2
    return nodes, weights

def midpt(m, lower=-1, upper=1):
    '''
    Midpoint rule:
    
    A well-known composite rule on the interval [lower, upper],
    corresponding to the composite Gauss rule with n = 1.
    The total number of evaluation points is m.
    '''
    h = (upper-lower)/m
    nodes = np.linspace(lower + h/2, upper - h/2, m)
    weights = np.full(m, h)
    return nodes, weights

def simps(m, lower=-1, upper=1):
    '''
    Simpson's rule:
    
    A third-order composite rule on the interval [lower, upper],
    corresponding to the Newton-Cotes rule with n = 3.
    The total number of evaluation points is 2*m + 1.
    '''
    h = (upper-lower)/m
    nodes = np.linspace(lower, upper, 2*m + 1)
    weights = np.empty(2*m + 1)
    weights[0] = weights[-1] = h/6
    weights[1::2] = 2/3*h
    weights[2:-1:2] = h/3
    return nodes, weights
    
def simps38(m, lower=-1, upper=1):
    '''
    Simpson's 3/8 rule:
    
    A third-order composite rule on the interval [lower, upper],
    corresponding to the Newton-Cotes rule with n = 4.
    Note that, while it is about twice as accurate as Simpson's rule,
    its order is the same. The total number of evaluation points is
    3*m + 1.
    '''
    h = (upper-lower)/m
    nodes = np.linspace(lower, upper, 3*m + 1)
    weights = np.empty(3*m + 1)
    weights[0] = weights[-1] = h/8
    weights[1::3] = 3/8*h
    weights[2::3] = 3/8*h
    weights[3:-1:3] = h/4
    return nodes, weights

def boole(m, lower=-1, upper=1):
    '''
    Boole's rule:
    
    A fifth-order composite rule on the interval [lower, upper],
    corresponding to the Newton-Cotes rule with n = 5.
    Boole's rule gains two orders on Simpson's 3/8 rule because
    its points are symmetrically distributed within each subinterval.
    The total number of evaluation points is 4*m + 1.
    '''
    h = (upper-lower)/m
    nodes = np.linspace(lower, upper, 4*m + 1)
    weights = np.empty(4*m + 1)
    weights[0] = weights [-1] = 7/90*h
    weights[1::4] = 16/45*h
    weights[2::4] = 2/15*h
    weights[3::4] = 16/45*h
    weights[4:-1:4] = 7/45*h
    return nodes, weights
    
def newton_cotes(m, n, lower=-1, upper=1):
    '''
    Newton-Cotes rules:
    
    These are the composite rules using n equally-spaced points in each
    subinterval. The order of a rule of this type is n-1 if n is even or
    n if n is odd. Rules with odd numbers of points gain an order because
    their points are equally distributed within each subinterval.
    High-order rules of this type can suffer from Runge's phenomenon.
    The total number of evaluation points is (n-1)*m + 1.
    '''
    h = (upper-lower)/m
    nodes = np.linspace(lower, upper, (n-1)*m + 1)
    vandermonde = np.stack(np.linspace(-1,1,n)**k for k in range(n))
    integrals = np.zeros(n)
    integrals[::2] = [1/(k+1) for k in range(0,n,2)]
    subinterval_weights = np.linalg.solve(vandermonde, integrals)
    weights = np.zeros((n-1)*m + 1)
    for i in range(m):
        weights[i*(n-1) : (i+1)*(n-1) + 1] += subinterval_weights*h
    return nodes, weights
    

def composite_gauss(m, n, lower=-1, upper=1):
    '''
    Composite Gauss rules:
    
    Composite rules of order 2*n-1 using Gauss-Legendre quadrature
    on each subinterval. The total number of evaluation points is n*m.
    '''
    h = (upper-lower)/m
    subinterval_nodes, subinterval_weights = leggauss(n)
    nodes = (h/2*np.tile(subinterval_nodes, m)
             + np.repeat(np.linspace(lower + h/2, upper - h/2, m), n))
    weights = np.tile(subinterval_weights*h/2, m)
    return nodes, weights
    

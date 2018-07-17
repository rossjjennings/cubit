import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt

def conprod_gauss():
    pass

def from_barycentric(nodes, weights, vertices):
    '''
    Compute nodes and weights of a cubature rule on a triangle with
    specified vertices from the barycentric coordinates of the nodes and 
    the relative weights. The triangle must lie in a plane.
    
    `nodes` should contain the barycentric coordinates of the nodes as 
            an n×3 array, where `n` is the number of nodes.
    `weights` should contain the `n` relative weights (summing to 1).
    `vertices` should be a 3×2 array with the vertices as rows. 
    
    Returns a tuple of an n×2 array whose columns are the nodes and
    a one-dimensional array of length n containing the weights
    (summing to the area of the triangle).
    '''
    R = np.array([np.concatenate((vertex, np.array([1.]))) for vertex in vertices])
    area = 1/2*np.linalg.det(R)
    return (nodes @ vertices).T, weights*area

def ac_6pt():
    '''
    Albrecht-Collatz 6-point rule (Stroud T2: 3-1):
    
    A third-degree rule with six points.
    
    Albrecht, J. and Collatz, L., "Zur numerischen auswertung mehrdimensionaler
    Integrale", Z. Agnew. Math. Mech., v. 38, 1958, pp. 1-15.
    '''
    r = 1/2
    u = 1/6
    v = 2/3
    B = 1/30
    C = 9/30
    
    nodes =  np.array([[r, r, s, u, u, v],
                       [r, s, r, u, v, u],
                       [s, r, r, v, u, u]])
    weights = np.array([B, B, B, C, C, C])
    
    return from_barycentric(nodes, weights, vertices)

def radon_7pt(vertices):
    '''
    Radon 7-point rule (Stroud T2: 5-1):
    
    A fifth-degree rule with seven points.
    
    Radon, J., "Zur mechanischen Kubatur",
    Monatsh. Math., v. 52, 1948, pp. 286-300.
    
    Hammer, P. C., Marlowe, O. J., and Stroud, A. H., 
    "Numerical integration over simplexes and cones",
    Math. Tables Aids Comput., v. 10, 1956, pp. 130-137.
    '''
    r = (6-sqrt(15))/21
    s = (9+2*sqrt(15))/21
    t = 1/3
    u = (6+sqrt(15))/21
    v = (9-2*sqrt(15))/21
    A = 9/40
    B = (155-sqrt(15))/1200
    C = (155+sqrt(15))/1200
    
    nodes =  np.array([[t, r, r, s, u, u, v],
                       [t, r, s, r, u, v, u],
                       [t, s, r, s, v, u, u]])
    weights = np.array([A, B, B, B, C, C, C])
    
    return from_barycentric(nodes, weights, vertices)

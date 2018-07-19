import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt

def conprod_gauss():
    pass

def from_barycentric(nodes, weights, vertices):
    '''
    Compute nodes and weights of a cubature rule on a triangle with
    specified vertices from the barycentric coordinates of the nodes and 
    the relative weights. The triangle need not lie in a plane.
    
    `nodes` should contain the barycentric coordinates of the nodes as 
            an n×3 array, where `n` is the number of nodes.
    `weights` should contain the `n` relative weights (summing to 1).
    `vertices` should be a 3×2 array with the vertices as rows. 
    
    Returns a tuple of an n×2 array whose columns are the nodes and
    a one-dimensional array of length n containing the weights
    (summing to the area of the triangle).
    '''
    sqnorms = np.sum(vertices**2, axis=1)
    cm_matrix = np.ones((4, 4)) # For computing Cayley-Menger determinant
    cm_matrix[-1,-1] = 0
    cm_matrix[:-1,:-1] = sqnorms + sqnorms[:,np.newaxis] - 2*vertices @ vertices.T
    area = 1/4*np.sqrt(-np.linalg.det(cm_matrix))
    return (nodes @ vertices).T, weights*area

def centroid(m=1):
    '''
    Centroid rule (Stroud Tn: 1-1 for n=2):
    
    The analog of the midpoint rule for triangles.
    A first-degree rule using only the value at the centroid.
    When tiled with m divisions per side, it uses m*(m+1)/2 points
    to integrate over the m*(m+1)/2 sub-triangles.
    '''
    

def wedge(m=1):
    '''
    Wedge rule (Stroud Tn: 1-2 for n=2):
    
    The analog of the trapezoid rule for triangles.
    A first-degree rule using the values at the vertices.
    When tiled with m divisions per side, it uses (m+1)*(m+2)/2 points
    to integrate over the m*(m+1)/2 sub-triangles.
    '''

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
    
    nodes =  np.array([[r, r, 0, u, u, v],
                       [r, 0, r, u, v, u],
                       [0, r, r, v, u, u]])
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

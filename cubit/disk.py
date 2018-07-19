import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt

def sphprod_gauss(n):

def square1():
    '''
    Square rule (Stroud S2: 3-1):
    
    A third-degree rule with four points at the vertices of a square.
    All points are inside the unit disk.
    '''
    x_nodes = np.array([ 1/2,    0, -1/2,    0])
    y_nodes = np.array([   0,  1/2,    0, -1/2])
    weights = np.array([ 1/4,  1/4,  1/4,  1/4])
    
    return (x_nodes, y_nodes), pi*weights

def square2():
    '''
    Inscribed square rule (Stroud S2: 3-2):
    
    A third-degree rule with four points at the vertices of a square.
    All points are on the boundary of the unit disk.
    '''
    r = 1/sqrt(2)
    
    x_nodes = np.array([   r,  -r,  -r,   r])
    y_nodes = np.array([   r,   r,  -r,  -r])
    y_nodes = np.array([ 1/4, 1/4, 1/4, 1/4])
    
    return x_nodes, y_nodes, pi*weights

def pentagon(alpha = 0):
    '''
    Pentagonal rule (Stroud S2: 4-1):
    
    A fourth-degree rule with six points. One point is at the origin and
    the other five are the vertices of a regular pentagon. Integrates with
    respect to the weight function w(x, y) = (x**2 + y**2)**(alpha/2).
    '''
    T = 2*pi/5
    r = sqrt((alpha+4)/(alpha+6))
    A = 4/(alpha+4)**2
    B = (alpha+2)*(alpha+6)/(5*(alpha+4)**2)
    V = 4*pi/(alpha+4)
    
    x_nodes = np.array([ 0, r, r*cos(T), r*cos(2*T), r*cos(3*T), r*cos(4*T)])
    y_nodes = np.array([ 0, 0, r*sin(T), r*sin(2*T), r*sin(3*T), r*sin(4*T)])
    weights = np.array([ A, B,        B,          B,          B,          B])
    
    return x_nodes, y_nodes, V*weights

def hexagon(alpha = 0):
    '''
    Hexagonal rule (Stroud S2: 5-1):
    
    A fifth-degree rule with seven points. One point is at the origin and
    the other six are at the vertices of a regular hexagon. Integrates with
    respect to the weight function w(x, y) = (x**2 + y**2)**(alpha/2).
    '''
    T = 2*pi/6
    r = sqrt((alpha+4)/(alpha+6))
    s = sqrt(3)/2
    A = 4/(alpha+4)**2
    B = (alpha+2)*(alpha+6)/(6*(alpha+4)**2)
    V = 4*pi/(alpha+4)
    
    x_nodes = np.array([ 0, r, r/2, -r/2, -r, -r/2, -r/2])
    y_nodes = np.array([ 0, 0, r*s,  r*s,  0, -r*s, -r*s])
    weights = np.array([ A, B,   B,    B,  B,    B,    B])
    
    return x_nodes, y_nodes, V*weights

def grid_9pt():
    '''
    Nine-point grid rule (Stroud S2: 5-2):
    
    A fifth-degree rule with nine points arranged in a 3×3 square grid.
    '''
    r = 1/sqrt(2)
    A = 1/6
    B = 1/24
    
    x_nodes = np.array([0,  r,  r,  0, -r, -r, -r,  0,  r])
    y_nodes = np.array([0,  0,  r,  r,  r,  0, -r, -r, -r])
    weights = np.array([A,  A,  B,  A,  B,  A,  B,  A,  B])
    
    return (x_nodes, y_nodes), pi*weights

def peirce_12pt():
    '''
    Peirce's twelve-point rule (Stroud S2: 7-1):
    
    A seventh-degree rule with twelve points. There is a typo in this
    formula as listed in Stroud: B2 is given as (551+4*sqrt(29))/6264.
    The correct value is (551+41*sqrt(29))/6264.
    
    Peirce, W. H., "Numerical Integration over Planar Regions",
    Ph.D. thesis, University of Wisconsin-Madison, 1956.
    
    Hammer, P. C., and Stroud, A. H., "Numerical evaluation of 
    multiple integrals II", Math. Tables Aids Comput., v. 11, 1957,
    pp. 59-67. MR 19, 323.
    '''
    r = sqrt(3/4)
    s = sqrt((27-3*sqrt(29))/104)
    t = sqrt((27+3*sqrt(29))/104)
    B1 = 2/27
    B2 = (551+41*sqrt(29))/6264
    B3 = (551-41*sqrt(29))/6264
    
    x_nodes = np.array([  r,  s,  t,  0, -s, -t, -r, -s, -t,  0,  s,  t])
    y_nodes = np.array([  0,  s,  t,  r,  s,  t,  0, -s, -t, -r, -s, -t])
    weights = np.array([ B1, B2, B3, B1, B2, B3, B1, B2, B3, B1, B2, B3])
    
    return (x_nodes, y_nodes), pi*weights

def double_octagon():
    '''
    Double octagon rule (Stroud S2: 7-2):
    
    A seventh-degree rule with 16 points at the vertices of two nested octagons.
    All weights are equal. This is also a spherical product Gauss rule. This
    version is rotated by π/8 compared to the version in Stroud.
    '''
    r = sqrt((3-sqrt(3))/6)
    s = sqrt((3+sqrt(3))/6)
    t = sqrt((3-sqrt(3))/12)
    u = sqrt((3+sqrt(3))/12)
    
    x_nodes = np.array([  r,  s,  t,  u,  0,  0, -t, -u,
                         -r, -s, -t, -u,  0,  0,  t,  u])
    y_nodes = np.array([  0,  s,  t,  u,  r,  s,  t,  u,
                          0, -s, -t, -u, -r, -s, -t, -u])
    weights = np.ones(16)/16
    
    return (x_nodes, y_nodes), pi*weights

def albrecht_19pt():
    '''
    Albrecht's nineteen-point rule (Stroud S2: 9-1):
    
    A ninth-degree rule with 19 points and order-6 rotational symmetry.
    
    Albrecht, J., "Formeln zur numerischen Integration uber Kreisbereiche",
    Z. Agnew. Math. Mech., v. 40, 1960, pp. 514-517. MR 22#11514.
    '''
    r1 = sqrt((96-4*sqrt(111))/155)
    r2 = sqrt((96+4*sqrt(111))/155)
    r3 = sqrt(4/5)
    s = sqrt(3)/2
    B0 = 251/2304
    B1 = (110297+5713*sqrt(111))/2045952
    B2 = (110297-5713*sqrt(111))/2045952
    C = 125/3072
    
    x_nodes = np.array([ 0,  r1,  r1/2, -r1/2,   -r1, -r1/2,  r1/2,
                             r2,  r2/2, -r2/2,   -r2, -r2/2,  r2/2,
                           r3*s,     0, -r3*s, -r3*s,     0,  r3*s])
    y_nodes = np.array([ 0,   0,  r1*s,  r1*s,     0, -r1*s, -r1*s,
                              0,  r2*s,  r2*s,     0, -r2*s, -r2*s,
                           r3/2,    r3,  r3/2, -r3/2,   -r3, -r3/2])
    weights = np.array([B0,  B1,    B1,    B1,    B1,    B1,    B1,
                             B2,    B2,    B2,    B2,    B2,    B2,
                              C,     C,     C,     C,     C,     C])
    
    return (x_nodes, y_nodes), pi*weights

def rr_20pt():
    pass

def lyusternik_21pt():
    pass

def rr_21pt():
    pass

def peirce_28pt():
    pass

def mysovskikh_28pt():
    pass

def albrecht_28pt():
    pass

def rr_28pt():
    pass

def peirce_32pt():
    pass

def rr_37pt():
    pass

def albrecht_41pt():
    pass

def mysovskikh_44pt():
    pass

def albrecht_48pt():
    pass

def albrecht_61pt():
    pass

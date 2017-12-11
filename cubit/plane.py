import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt
from cubit import line

def prod_hermgauss(n1, n2):
    '''
    Product Gauss-Hermite rule:
    
    The product form of a Gauss-Hermite quadrature rule.
    If n1 == n2 == n, this is a rule of order 2*n-1, using n**2 points.
    '''
    nodes1, weights1 = line.hermgauss(n1)
    nodes2, weights2 = line.hermgauss(n2)
    x_nodes = np.tile(nodes1, n2)
    y_nodes = np.repeat(nodes2, n1)
    weights = np.tile(weights1, n2) * np.repeat(weights2, n1)
    return (x_nodes, y_nodes), weights

def pentagon():
    '''
    Pentagonal rule (Stroud Er22: 4-1):
    
    A fourth-degree rule with six points. One point is at the origin and
    the other five are the vertices of a regular pentagon. Integrates with
    respect to the weight function w(x, y) = exp(-x**2-y**2).
    '''
    T = 2*pi/5
    x_nodes = np.array([0, 1, cos(T), cos(2*T), cos(3*T), cos(4*T)])
    y_nodes = np.array([0, 0, sin(T), sin(2*T), sin(3*T), sin(4*T)])
    x_nodes *= sqrt(2)
    y_nodes *= sqrt(2)
    weights = np.array([1/2, 1/10, 1/10, 1/10, 1/10, 1/10])
    weights *= pi
    return (x_nodes, y_nodes), weights

def hexagon():
    '''
    Hexagonal rule (Stroud Er22: 5-1):
    
    A fifth-degree rule with seven points. One point is at the origin and
    the other six are the vertices of a regular hexagon. Integrates with
    respect to the weight function w(x, y) = exp(-x**2-y**2).
    
    Stroud, A. H., and Secrest, D., "Approximate integration formulas for
    certain spherically symmetric regions", Math. Comput., v. 17, 1963,
    pp. 105-135.
    '''
    r = sqrt(2)
    s = sqrt(1/2)
    t = sqrt(3/2)
    x_nodes = np.array([0, r, s, -s, -r, -s,  s])
    y_nodes = np.array([0, 0, t,  t,  0, -t, -t])
    weights = np.array([1/2, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12])
    weights *= pi
    return (x_nodes, y_nodes), weights

def ss_12pt():
    '''
    Stroud-Secrest twelve-point rule (Stroud Er22: 7-1):
    
    A seventh-degree rule with twelve points. Integrates with respect to the
    weight function w(x, y) = exp(-x**2-y**2).
    
    Stroud, A. H., and Secrest, D., "Approximate integration formulas for
    certain spherically symmetric regions", Math. Comput., v. 17, 1963,
    pp. 105-135.
    '''
    r = sqrt(3)
    s = sqrt((9-3*sqrt(5))/8)
    t = sqrt((9+3*sqrt(5))/8)
    A = 1/36
    B = (5+2*sqrt(5))/45
    C = (5-2*sqrt(5))/45
    
    x_nodes = np.array([r, 0, -r, 0, s, -s, -s, s, t, -t, -t, t])
    y_nodes = np.array([0, r, 0, -r, s, s, -s, -s, t, t, -t, -t])
    weights = np.array([A, A, A,  A, B, B,  B,  B, C, C,  C,  C])
    weights *= pi
    
    return (x_nodes, y_nodes), weights

def rr_20pt():
    '''
    Rabinowitz-Richter 20-point rule (Stroud Er22: 9-1):
    
    A ninth-degree rule with 20 points. Integrates with respect to the
    weight function w(x, y) = exp(-x**2-y**2).
    
    Rabinowitz, P., and Richter, N., "Perfectly symmetric two-dimensional
    integration formulas with minimal numbers of points", Math. Comput.,
    v. 23, 1969, pp. 765-779.
    '''
    r1 = 1.538189001320852 # == sqrt((3 + sqrt(3))/2)
    r2 = 1.224744871391589 # == sqrt(3/2)
    r3 = 0.4817165220011443 # == sqrt((2*sqrt(3) - 3)/2)
    r4 = 2.607349811958554 # == sqrt((6 + sqrt(3) + sqrt(24 + 6*sqrt(3)))/2)
    s4 = 0.9663217712794149 # == sqrt((6 + sqrt(3) - sqrt(24 + 6*sqrt(3)))/2)
    B1 = 0.1237222328857347 # == (5 - 2*sqrt(3))/39*pi
    B2 = 0.06544984694978697 # == pi/48
    B3 = 0.5935280476180875 # == (59 + 34*sqrt(3))/624*pi
    B4 = 0.001349017971918148 # == (2 - sqrt(3))/624*pi
    
    x_nodes = np.array([ r1,  r4,  r3,  r2,  s4,   0, -s4, -r3, -r2, -r4,
                        -r1, -r4, -r3, -r2, -s4,   0,  s4,  r3,  r2,  r4])
    y_nodes = np.array([  0,  s4,  r3,  r2,  r4,  r1,  r4,  r3,  r2,  s4,
                          0, -s4, -r3, -r2, -r4, -r1, -r4, -r3, -r2, -s4])
    weights = np.array([ B1,  B4,  B3,  B2,  B4,  B1,  B4,  B3,  B2,  B4,
                         B1,  B4,  B3,  B2,  B4,  B1,  B4,  B3,  B2,  B4])
    
    return (x_nodes, y_nodes), weights

def rr_28pt():
    '''
    First Rabinowitz-Richter 28-point rule (Stroud Er22: 11-1):
    
    An eleventh-degree rule with 28 points. Integrates with respect to the
    weight function w(x, y) = exp(-x**2-y**2).
    
    Rabinowitz, P., and Richter, N., "Perfectly symmetric two-dimensional
    integration formulas with minimal numbers of points", Math. Comput.,
    v. 23, 1969, pp. 765-779.
    '''
    r1 = 2.757816396257008 # == sqrt(4 + sqrt(13))
    r2 = 1.732050807568877 # == sqrt(3)
    r3 = 0.6280515301597559 # == sqrt(4 - sqrt(13))
    r4 = 1.224744871391589 # == sqrt(3/2)
    s4 = 2.121320343559643 # == sqrt(9/2)
    r5 = 0.7071067811865475 # == sqrt(1/2)
    s5 = 1.224744871391589 # == sqrt(3/2)
    B1 = 8.176645817675417e-4 # == (481 - 133*sqrt(13))*pi/5616
    B2 = 4.363323129985824e-2 # == pi/72
    B3 = 0.5373255214498174 # == (481 + 133*sqrt(13))*pi/5616
    B4 = 3.636102608321520e-3 # == pi/864
    B5 = 9.817477042468103e-2 # == pi/32
    
    x_nodes = np.array([ r3,  r2,  r1,  s5,  s4,  r5,  r4,
                          0,   0,   0, -r5, -r4, -s5, -s4,
                        -r3, -r2, -r1, -s5, -s4, -r5, -r4,
                          0,   0,   0,  s5,  s4,  s5,  s4])
    y_nodes = np.array([  0,   0,   0,  r5,  r4,  s5,  s4,
                         r3,  r2,  r1,  s5,  s4,  r5,  r4,
                          0,   0,   0, -r5, -r4, -s5, -s4,
                        -r3, -r2, -r1, -s5, -s4, -r5, -r4])
    weights = np.array([ B3,  B2,  B1,  B5,  B4,  B5,  B4,
                         B3,  B2,  B1,  B5,  B4,  B5,  B4,
                         B3,  B2,  B1,  B5,  B4,  B5,  B4,
                         B3,  B2,  B1,  B5,  B4,  B5,  B4])
    
    return (x_nodes, y_nodes), weights

def rr_28pt2():
    '''
    Second Rabinowitz-Richter 28-point rule (Stroud Er22: 11-2):
    
    An eleventh-degree rule with 28 points. Integrates with respect to the
    weight function w(x, y) = exp(-x**2-y**2).
    
    Rabinowitz, P., and Richter, N., "Perfectly symmetric two-dimensional
    integration formulas with minimal numbers of points", Math. Comput.,
    v. 23, 1969, pp. 765-779.
    '''
    r1 = 2.907364117106118
    r2 = 1.528230917660483
    r3 = 0.6178819071436261
    r4 = 1.904162039910276
    r5 = 0.9724173472297303
    r6 = 2.061552812808830
    s6 = 0.8660254037844387
    B1 = 4.106569066965604e-4
    B2 = 9.065690889492120e-2
    B3 = 0.5266955729327722
    B4 = 9.681125175723808e-4
    B5 = 0.1515812331366514
    B6 = 7.542839504417270e-3
    
    x_nodes = np.array([ r1,  r2,  r3,  r6,  r5,  r4,  s6, 
                          0,   0,   0, -s6, -r5, -r4, -r6,
                        -r1, -r2, -r3, -r6, -r5, -r4, -s6,
                          0,   0,   0,  s6,  r5,  r4,  r6])
    y_nodes = np.array([  0,   0,   0,  s6,  r5,  r4,  r6,
                         r1,  r2,  r3,  r6,  r5,  r4,  s6,
                          0,   0,   0, -s6, -r5, -r4, -r6,
                        -r1, -r2, -r3, -r6, -r5, -r4, -s6])
    weights = np.array([ B1,  B2,  B3,  B6,  B5,  B4,  B6,
                         B1,  B2,  B3,  B6,  B5,  B4,  B6,
                         B1,  B2,  B3,  B6,  B5,  B4,  B6,
                         B1,  B2,  B3,  B6,  B5,  B4,  B6])
    
    return (x_nodes, y_nodes), weights

def rr_37pt():
    '''
    Rabinowitz-Richter 37-point rule (Stroud Er22: 13-1):
    
    A thirteenth-degree rule with 37 points. One weight is negative.
    Integrates with respect to the weight function w(x, y) = exp(-x**2-y**2). 
    
    Rabinowitz, P., and Richter, N., "Perfectly symmetric two-dimensional
    integration formulas with minimal numbers of points", Math. Comput.,
    v. 23, 1969, pp. 765-779.
    '''
    r1 = 2.403151765001966
    r2 = 1.298479973315986
    r3 = 1.912428205769905
    r4 = 0.9478854439698223
    r5 = 0.3188824732576547
    r6 = 3.325657829663178
    s6 = 1.145527285699371
    r7 = 1.882228401823884
    s7 = 0.8826073082889659
    A = -0.7482913219380363
    B1 = 3.521509661098668e-3
    B2 = 0.1650055872539264
    B3 = 8.537825937946404e-4
    B4 = 0.1326938806789336
    B5 = 0.6447719928481539
    B6 = 1.799266413507747e-5
    B7 = 1.279412775888998e-2
    
    x_nodes = np.array([0,  r1,  r2,  r6,  r7,  r3,  r4,  r5,  s7,  s6,
                             0,   0, -s6, -s7, -r3, -r4, -r5, -r7, -r6,
                           -r1, -r2, -r6, -r7, -r3, -r4, -r5, -s7, -s6,
                             0,   0,  s6,  s7,  r3,  r4,  r5,  r7,  r6])
    y_nodes = np.array([0,   0,   0,  s6,  s7,  r3,  r4,  r5,  r7,  r6,
                            r1,  r2,  r6,  r7,  r3,  r4,  r5,  s7,  s6,
                             0,   0, -s6, -s7, -r3, -r4, -r5, -r7, -r6,
                           -r1, -r2, -r6, -r7, -r3, -r4, -r5, -s7, -s6])
    weights = np.array([A,  B1,  B2,  B6,  B7,  B3,  B4,  B5,  B7,  B6,
                            B1,  B2,  B6,  B7,  B3,  B4,  B5,  B7,  B6,
                            B1,  B2,  B6,  B7,  B3,  B4,  B5,  B7,  B6,
                            B1,  B2,  B6,  B7,  B3,  B4,  B5,  B7,  B6])
    
    return (x_nodes, y_nodes), weights

def rr_44pt():
    '''
    Rabinowitz-Richter 44-point rule (Stroud Er22: 15-1):
    
    A fifteenth-degree rule with 44 points. Integrates with respect to the
    weight function w(x, y) = exp(-x**2-y**2).
    
    Rabinowitz, P., and Richter, N., "Perfectly symmetric two-dimensional
    integration formulas with minimal numbers of points", Math. Comput.,
    v. 23, 1969, pp. 765-779.
    '''
    r1 = 3.538388728121807
    r2 = 2.359676416877929
    r3 = 1.312801844620926
    r4 = 0.5389559482114205
    r5 = 2.300279949805658
    r6 = 1.581138830084189
    r7 = 0.8418504335819279
    r8 = 2.685533581755341
    s8 = 1.112384431771456
    r9 = 1.740847514397403
    s9 = 0.7210826504868960
    B1 = 8.006483569659628e-6
    B2 = 3.604577420838264e-3
    B3 = 0.1187609330759137
    B4 = 0.4372488543791402
    B5 = 3.671735075832989e-5
    B6 = 5.654866776461627e-3
    B7 = 0.1777774268424240
    B8 = 2.735449647853290e-4
    B9 = 2.087984556938594e-2
    
    x_nodes = np.array([ r1,  r2,  r3,  r4,  r5,  r6,  r7,  r8, -r8,  r9, -r9,
                          0,   0,   0,   0, -r5, -r6, -r7,  s8, -s8,  s9, -s9,
                        -r1, -r2, -r3, -r4, -r5, -r6, -r7, -s8,  s8, -s9,  s9,
                          0,   0,   0,   0,  r5,  r6,  r7, -r8,  r8, -r9,  r9])
    
    y_nodes = np.array([  0,   0,   0,   0,  r5,  r6,  r7,  s8, -s8,  s9, -s9,
                         r1,  r2,  r3,  r4,  r5,  r6,  r7,  r8, -r8,  r9, -r9,
                          0,   0,   0,   0, -r5, -r6, -r7,  r8, -r8,  r9, -r9,
                        -r1, -r2, -r3, -r4, -r5, -r6, -r7,  s8, -s8,  s9, -s9])
    
    weights = np.array([ B1,  B2,  B3,  B4,  B5,  B6,  B7,  B8,  B8,  B9,  B9,
                         B1,  B2,  B3,  B4,  B5,  B6,  B7,  B8,  B8,  B9,  B9,
                         B1,  B2,  B3,  B4,  B5,  B6,  B7,  B8,  B8,  B9,  B9,
                         B1,  B2,  B3,  B4,  B5,  B6,  B7,  B8,  B8,  B9,  B9])
    
    return (x_nodes, y_nodes), weights


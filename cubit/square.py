import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt
from cubit import interval

def prod_trapz(m1, m2):
    '''
    Product trapezoid rule (Stroud C2: 1-5):
    
    The product form of the trapezoid rule. 
    This is a first-order rule with four points.
    When tiled with m1 points on one side and m2 on the other,
    it uses a total of (m1+1)*(m2+1) points.
    '''
    hx, hy = 2/m1, 2/m2
    x_nodes = np.tile(np.linspace(-1, 1, m1+1), m2+1)
    y_nodes = np.repeat(np.linspace(-1, 1, m2+1), m1+1)
    weights = np.full((m1+1)*(m2+1), h*k)
    weights[0] = weights[m1] = weights[-m1-1] = weights[-1] = hx*hy/4
    weights[1:m1] = weights[-m1:-1] = np.full(m1-1, hx*hy/2)
    weights[m1+1:-m1-1:m1+1] = np.full(m2-1, hx*hy/2)
    weights[2*m1+1:-m1-2:m1+1] = np.full(m2-1, hx*hy/2)
    return (x_nodes, y_nodes), weights
    

def prod_simps(m1, m2):
    '''
    Product Simpson's rule (Stroud C2: 3-3):
    
    The product form of Simpson's rule.
    This is a third-order rule with nine points.
    '''
    nodes_x, weights_x = interval.simps(m1)
    nodes_y, weights_y = interval.simps(m2)
    x_nodes = np.tile(nodes_x, m2)
    y_nodes = np.repeat(nodes_y, m1)
    weights = np.tile(weights_x, m2) * np.repeat(weights_y, m1)
    return (x_nodes, y_nodes), weights

def prod_gauss(n1, n2):
    '''
    Product Gauss rule (Stroud C2: 3-1, 5-4, 7-4):
    
    The product form of a Gauss-Legendre quadrature rule.
    If n1 == n2 == n, this is a rule of order 2*n-1, using n**2 points.
    '''
    nodes1, weights1 = interval.leggauss(n1)
    nodes2, weights2 = interval.leggauss(n2)
    x_nodes = np.tile(nodes1, n2)
    y_nodes = np.repeat(nodes2, n1)
    weights = np.tile(weights1, n2) * np.repeat(weights2, n1)
    return (x_nodes, y_nodes), weights

def prod_newton_cotes(m1, m2, n1, n2):
    '''
    Product Newton-Cotes rule:
    
    The product form of an arbitrary Newton-Cotes rule.
    If n is even, this rule has order n-1; if n is odd, it has order n.
    It uses n**2 points.
    '''
    nodes1, weights1 = interval.newton_cotes(m1, n1)
    nodes2, weights2 = interval.newton_cotes(m2, n2)
    x_nodes = np.tile(nodes1, m2)
    y_nodes = np.repeat(nodes2, m1)
    weights = np.tile(weights1, m2) * np.repeat(weights2, m1)
    return (x_nodes, y_nodes), weights

def ewing_quincunx(m1, m2):
    '''
    Ewing's quincuncial rule (Stroud C2: 3-2):
    
    A third-order rule on the square using five points arranged in a quincunx.
    This is a special case of Ewing's rule for hypercubes. Four of the points
    are on the boundary, so when tiled with m1 subdivisions on one side and m2 
    on the other, it uses only 2*m1*m2 + m1 + m2 + 1 points total.
    
    Ewing, G. M., "On approximate cubature",
    Amer. Math. Monthly, v. 4, 1941, pp. 134-136.
    '''
    longrow = np.linspace(-1, 1, m1 + 1)
    shortrow = np.linspace(-1 + 1/m1, 1 - 1/m1, m1)
    nodes_x = np.empty((m1 + 1)*(m2 + 1) + m1*m2)
    nodes_y = np.empty((m1 + 1)*(m2 + 1) + m1*m2)
    weights = np.empty((m1 + 1)*(m2 + 1) + m1*m2)
    for i in range(m2):
        nodes_x[(2*m1 + 1)*i : (2*m1 + 1)*i + m1 + 1] = longrow
        nodes_y[(2*m1 + 1)*i : (2*m1 + 1)*i + m1 + 1] = longrow[i]
        weights[(2*m1 + 1)*i] = weights[(2*m1 + 1)*i + m1] = 1/6
        weights[(2*m1 + 1)*i + 1 : (2*m1 + 1)*i + m1] = 1/3
        nodes_x[(2*m1 + 1)*i + m1 + 1 : (2*m1 + 1)*(i + 1)] = shortrow
        nodes_y[(2*m1 + 1)*i + m1 + 1 : (2*m1 + 1)*(i + 1)] = shortrow[i]
        weights[(2*m1 + 1)*i + m1 + 1 : (2*m1 + 1)*(i + 1)] = 2/3
    weights[0 : m1 + 1] *= 1/2
    nodes_x[(2*m1 + 1)*m2:] = longrow
    nodes_y[(2*m1 + 1)*m2:] = longrow[-1]
    weights[(2*m1 + 1)*m2] = weights[-1] = 1/12
    weights[(2*m1 + 1)*m2 + 1 : -1] = 1/6
    weights *= 4/(m1*m2)
    return (nodes_x, nodes_y), weights

def ac_9pt(m1, m2):
    '''
    Albrecht-Collatz nine-point rule (Stroud C2: 3-4):
    
    A third-order rule with nine points, eight of which are on the boundary.
    When tiled with m1 subdivisions on one side and m2 on the other, it uses
    4*m1*m2 + 2*m1 + 2*m2 + 1 points.
    
    Albrecht, J. and Collatz, L., "Zur numerischen auswertung mehrdimensionaler
    Integrale", Z. Agnew. Math. Mech., v. 38, 1958, pp. 1-15.
    '''
    pass

def irwin_12pt():
    '''
    Irwin's twelve-point rule (Stroud C2: 3-5):
    
    A third-order rule with twelve points, eight of which are outside the
    region and have negative weights. The remaining four points are at
    the corners.
    
    Irwin, J. O., "On quadrature and cubature",
    Tracts for Computers, No. 10, 1923.
    '''
    pass

def radon_7pt():
    '''
    Radon's seven-point rule (Stroud C2: 5-1):
    
    A fifth-order rule with seven points. The points are distributed
    symmetrically with respect to the x- and y-axes. Stroud notes that
    the points are the common zeros of the orthogonal polynomials
        x**3 - 3/5*x,  x*y**2 - 1/3*x, and x**2*y + y**2 - 14/15*y.
    Similar fifth-order rules with the same number of points were constructed
    by Radon for other regions, and are also the common zeros of triples of
    orthogonal polynomials.
    
    Radon, J., "Zur mechaniste Kubatur",
    Monatsh. Math., v. 52, 1948, pp. 286-300.
    '''
    r = sqrt(3/5)
    s = sqrt(1/3)
    t = sqrt(14/15)
    x_nodes = np.array([0, r, 0, -r, -r,  0,  r]) 
    y_nodes = np.array([0, s, t,  s, -s, -t, -s])
    weights = np.array([2/7, 5/36, 5/63, 5/36, 5/36, 5/63, 5/36])
    weights *= 4
    return (x_nodes, y_nodes), weights

def ac_7pt():
    '''
    Albrecht-Collatz seven-point rule (Stroud C2: 5-2):
    
    A fifth-order fule with seven points. The points are distributed
    symmetrically with respect to the diagonal.
    
    Albrecht, J. and Collatz, L., "Zur numerischen auswertung mehrdimensionaler
    Integrale", Z. Agnew. Math. Mech., v. 38, 1958, pp. 1-15.
    '''
    r = sqrt(7/15)
    s = sqrt((7+sqrt(24))/15)
    t = sqrt((7-sqrt(24))/15)
    x_nodes = np.array([0, r, -t, -s, -r,  s,  t])
    y_nodes = np.array([0, r,  s,  t, -r, -t, -s])
    weights = np.array([2/7, 25/168, 5/48, 5/48, 25/168, 5/48, 5/48])
    weights *= 4
    return (x_nodes, y_nodes), weights

def burnside_8pt():
    '''
    Burnside's eight-point rule (Stroud C2: 5-3):
    
    A fifth-order rule with eight points, which have the same symmetry
    as the square. The points are the common zeros of the orthogonal
    polynomials
        x**3 - 2/5*x*y**2 - 7/15*x,  y**3 - 2/5*x**2*y - 7/15*y,
             and x**2*y**2 - 5/9*x**2 - 5/9*y**2 + 7/27.
    
    Burnside, W., "An approximate quadrature formula", Messenger of Math.,
    v. 37, 1908, pp. 166-167
    '''
    x_nodes = np.array([sqrt(7/15), sqrt(7/9), 0, -sqrt(7/9),
                        -sqrt(7/15), -sqrt(7/9), 0, sqrt(7/9)])
    y_nodes = np.array([0, sqrt(7/9), sqrt(7/15), sqrt(7/9),
                        0, -sqrt(7/9), -sqrt(7/15), -sqrt(7/9)])
    weights = np.array([10/49, 9/196, 10/49, 9/196,
                        10/49, 9/196, 10/49, 9/196])
    weights *= 4
    return (x_nodes, y_nodes), weights

def tyler_13pt():
    '''
    Tyler's thirteen-point rule (Stroud C2: 5-5):
    
    A fifth-order rule with thirteen points. The center has negative weight.
    Four points lie at the corners. The points are the common zeros of the
    orthogonal polynomials
        x**3*y - xy and x*y**3 - xy.
    
    Tyler, G. W., "Numerical integration of functions of several variables",
    Canad. J. Math., v. 5, 1953, pp. 393-412.
    '''
    x_nodes = np.array([0, 0.5,  1,  1,    0,  0, -1,
                          -0.5, -1, -1,    0,  0,  1])
    y_nodes = np.array([0,   0,  0,  1,  0.5,  1,  1,
                             0,  0, -1, -0.5, -1, -1])
    weights = np.array([-28/45, 16/45, 1/45, 1/36, 16/45, 1/45, 1/36,
                                16/45, 1/45, 1/36, 16/45, 1/45, 1/36])
    weights *= 4
    return (x_nodes, y_nodes), weights

def meister_13pt():
    '''
    Meister's thirteen-point rule (Stroud C2: 5-6):
    
    A fifth-order rule with thirteen points. Four points lie at corners,
    and four lie along edges.
    
    Meister, Bernd, "On a family of cubature formulae",
    Computer J., v. 8, 1966, pp. 368-371.
    '''
    x_nodes = np.array([0, 1,  0.5,  1,  0, -0.5, -1,
                          -1, -0.5, -1,  0,  0.5,  1])
    y_nodes = np.array([0, 0,  0.5,  1,  1,  0.5,  1,
                           0, -0.5, -1, -1, -0.5, -1])
    weights = np.array([2/45, 2/45, 8/45, 1/60, 2/45, 8/45, 1/60,
                              2/45, 8/45, 1/60, 2/45, 8/45, 1/60])
    weights *= 4
    return (x_nodes, y_nodes), weights

def irwin_24pt():
    '''
    Irwin's 24-point rule (Stroud C2: 5-7):
    
    A fifth-order rule with 24 points. Eight points have negative weights.
    Four lie at the corners, and the rest are outside the region.
    
    Irwin, J. O., "On quadrature and cubature",
    Tracts for Computers, No. 10, 1923.
    '''
    x_nodes = np.array([ 5,   3,   1,  3,   1,  1,
                        -1,  -1,  -1, -3,  -3, -5,
                        -5,  -3,  -1, -3,  -1, -1,
                         1,   1,   1,  3,   3,  5])
    y_nodes = np.array([ 1,   1,   1,  3,   3,  5,
                         5,   3,   1,  3,   1,  1,
                        -1,  -1,  -1, -3,  -3, -5,
                        -5,  -3,  -1, -3,  -1, -1])
    weights = np.array([11, -98, 889,  5, -98, 11,
                        11, -98, 889,  5, -98, 11,
                        11, -98, 889,  5, -98, 11,
                        11, -98, 889,  5, -98, 11])
    weights *= 4/2880
    return (x_nodes, y_nodes), weights

def tyler_12pt():
    '''
    Tyler's twelve-point rule (Stroud C2: 7-1):
    
    A seventh-order rule with twelve points. The points are the common zeros
    of the orthogonal polynomials
        x**3*y - x*y**3,    x**4 - y**4 - 6/7*x**2 + 6/7*y**2,
        and x**4 + 54/55*x**2*y**2 + y**4 - 456/385*(x**2 + y**2) + 108/385.
    
    Tyler, G. W., "Numerical integration of functions of several variables",
    Canad. J. Math., v. 5, 1953, pp. 393-412.
    '''
    r = sqrt(6/7)
    s = sqrt((114-3*sqrt(583))/287)
    t = sqrt((114+3*sqrt(583))/287)
    B1 = 49/810
    B2 = (178981+2769*sqrt(583))/1888920
    B3 = (178981-2769*sqrt(583))/1888920
    x_nodes = np.array([ r,  s,  t, 0, -s, -t,
                        -r, -s, -t, 0,  s,  t])
    y_nodes = np.array([0,  s,  t,  r,  s,  t,
                        0, -s, -t, -r, -s, -t])
    weights = np.array([B1, B2, B3, B1, B2, B3,
                        B1, B2, B3, B1, B2, B3])
    return (x_nodes, y_nodes), weights

def mysovskikh_12pt():
    '''
    Mysovskikh-Phillips twelve-point rule (Stroud C2: 7-2):
    
    A seventh-order rule with twelve points. Four points lie outside
    the square. The points are the common zeros of the orthogonal polynomials
        x**3*y - 3/5*x*y,   x*y**3 - 3/5*x*y,
        and x**4 + 27/14*x**2*y**2 + y**4 - 3/2*x**2 - 3/2*y**2 + 27/70.
    
    Mysovskikh, I. P., "On the construction of cubature formulas for the
    simplest regions" (in Russian), Metody Vyčisl., v. 1, 1963, pp. 3-11.
    
    Phillips, G. M., "Numerical integration in two and three dimensions",
    Computer J., v. 10, 1967, pp. 202-204.
    '''
    r = sqrt((105 + 3*sqrt(385))/140)
    s = sqrt((105 - 3*sqrt(385))/140)
    t = sqrt(3/5)
    B1 = (77 - 3*sqrt(385))/891
    B2 = (77 + 3*sqrt(385))/891
    B3 = 25/324
    
    x_nodes = np.array([ r,  s,  t,  0,  0, -t,
                        -r, -s, -t,  0,  0,  t])
    y_nodes = np.array([ 0,  0,  t,  r,  s,  t,
                         0,  0, -t, -r, -s, -t])
    weights = np.array([B1, B2, B3, B1, B2, B3,
                        B1, B2, B3, B1, B2, B3])
    weights *= 4
    
    return (x_nodes, y_nodes), weights

def maxwell_13pt():
    '''
    Maxwell's thirteen-point rule (Stroud C2: 7-3):
    
    A seventh-order rule with thirteen points. The points are the common
    zeros of the orthogonal polynomials
        x**3*y + x*y**3 - 6/5*x*y and
        x**4 - 54/35*x**2*y**2 + y**4 - 12/35*x**2 - 12/35*y**2.
    
    Maxwell, J. Clerk, "On approximate multiple integration between limits
    of summation", Proc. Cambridge Philos. Soc., v. 3, 1877, pp. 39-47.
    '''
    r = sqrt(12/35)
    s = sqrt((93 + 3*sqrt(186))/155)
    t = sqrt((93 - 3*sqrt(186))/155)
    
    x_nodes = np.array([0,  r,  s,  t,  0, -t, -s,
                           -r, -s, -t,  0,  t,  s])
    y_nodes = np.array([0,  0,  t,  s, -r, -s, -t,
                            0, -t, -s, -r, -s, -t])
    weights = np.array([1/81, 49/324, 31/649, 31/649, 49/324, 31/649, 31/649,
                              49/324, 31/649, 31/649, 49/324, 31/649, 31/649])
    weights *= 4
    
    return (x_nodes, y_nodes), weights

def tyler_21pt():
    '''
    Tyler's 21-point rule (Stroud C2: 7-5):
    
    A seventh-order rule with 21 points. Four points have negative weight.
    The formula is incorrectly listed as having 12 points in Stroud.
    
    Tyler, G. W., "Numerical integration of functions of several variables",
    Canad. J. Math., v. 5, 1953, pp. 393-412.
    '''
    x_nodes = np.array([0,  1,  2/3,  1/3,  1,  1/2,
                            0,    0,    0, -1, -1/2,
                           -1, -2/3, -1/3, -1, -1/2,
                            0,    0,    0,  1,  1/2])
    y_nodes = np.array([0,  0,    0,    0,  1,  1/2,
                            1,  2/3,  1/3,  1,  1/2,
                            0,    0,    0, -1, -1/2,
                           -1, -2/3, -1/3, -1, -1/2])
    weights = np.array([449/315, 37/1260, 3/28, -69/140, 7/540, 32/135,
                                 37/1260, 3/28, -69/140, 7/540, 32/135,
                                 37/1260, 3/28, -69/140, 7/540, 32/135,
                                 37/1260, 3/28, -69/140, 7/540, 32/135])
    weights *= 4
    
    return (x_nodes, y_nodes), weights

def meister_25pt():
    '''
    Meister's 25-point rule (Stroud C2: 7-6):
    
    A seventh-order rule with 25 points. Four points have negative weight.
    
    Meister, Bernd, "On a family of cubature formulae",
    Computer J., v. 8, 1966, pp. 368-371.
    '''
    x_nodes = np.array([   0,  2/3,    1,  1,  2/3,  1/3,  1/3,
                                 0, -1/3, -1, -2/3, -1/3,   -1,
                              -2/3,   -1, -1, -2/3, -1/3, -1/3,
                                 0,  1/3,  1   2/3,  1/3,  1/3])
    y_nodes = np.array([   0,    0,  1/3,  1,  2/3,  1/3,    1,
                               2/3,    1,  1,  2/3,  1/3,  1/3,
                                 0, -1/3, -1, -2/3, -1/3,   -1,
                              -2/3,   -1, -1, -2/3, -1/3, -1/3])
    weights = np.array([1024,  576,  117, 47,  576,   -9,  117,
                               576,  117, 47,  576,   -9,  117,
                               576,  117, 47,  576,   -9,  117,
                               576,  117, 47,  576,   -9,  117])
    weights *= 4/6720
    
    return (x_nodes, y_nodes), weights

def rr_20pt():
    '''
    Rabinowitz-Richter 20-point rule (Stroud C2: 9-1):
    
    A ninth-order rule with 20 points.
    
    Rabinowitz, P., and Richter, N., "Perfectly symmetric two-dimensional
    integration formulas with minimal numbers of points", Math. Comput.,
    v. 23, 1969, pp. 765-779.
    '''
    pass

def chanut_21pt():
    '''
    Chanut's 21-point rule (Stroud C2: 9-2):
    
    A ninth-order rule with 21 points.
    
    Chanut, A., "Calcul numérique des integrales doubles",
    C. R. Acad. Sci. Paris, v. 256, 1963, pp. 3239-3241.
    '''
    pass

def chanut_25pt():
    '''
    Chanut's 25-point rule (Stroud C2: 9-3):
    
    A ninth-order rule with 25 points.
    
    Chanut, A., "Calcul numérique des integrales doubles",
    C. R. Acad. Sci. Paris, v. 256, 1963, pp. 3239-3241.
    '''
    pass

def rr_25pt():
    '''
    Rabinowitz-Richter 25-point rule (Stroud C2: 11-1):
    
    An eleventh-order rule with 25 points. Four points are outside the square.
    
    Rabinowitz, P., and Richter, N., "Perfectly symmetric two-dimensional
    integration formulas with minimal numbers of points", Math. Comput.,
    v. 23, 1969, pp. 765-779.
    '''
    pass

def rr_28pt():
    '''
    Rabinowitz-Richter 28-point rule (Stroud C2: 11-2):
    
    An eleventh-order rule with 28 points.
    
    Rabinowitz, P., and Richter, N., "Perfectly symmetric two-dimensional
    integration formulas with minimal numbers of points", Math. Comput.,
    v. 23, 1969, pp. 765-779.
    '''
    pass

def rr_37pt():
    '''
    Rabinowitz-Richter 37-point rule (Stroud C2: 13-1):
    
    A thirteenth-order rule with 37 points.
    
    Rabinowitz, P., and Richter, N., "Perfectly symmetric two-dimensional
    integration formulas with minimal numbers of points", Math. Comput.,
    v. 23, 1969, pp. 765-779.
    '''
    pass

def rr_44pt():
    '''
    Rabinowitz-Richter 44-point rule (Stroud C2: 15-1):
    
    A fifteenth-order rule with 44 points. Four points are outside the square.
    
    Rabinowitz, P., and Richter, N., "Perfectly symmetric two-dimensional
    integration formulas with minimal numbers of points", Math. Comput.,
    v. 23, 1969, pp. 765-779.
    '''
    pass

def rr_48pt():
    '''
    Rabinowitz-Richter 48-point rule (Stroud C2: 15-2):
    
    A fifteenth-order rule with 48 points.
    
    Rabinowitz, P., and Richter, N., "Perfectly symmetric two-dimensional
    integration formulas with minimal numbers of points", Math. Comput.,
    v. 23, 1969, pp. 765-779.
    '''
    pass

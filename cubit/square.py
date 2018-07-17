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
    
    Radon, J., "Zur mechanischen Kubatur",
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
    r1 = 0.9845398119422523
    r2 = 0.4888863428423724
    r3 = 0.9395672874215217
    r4 = 0.8367103250239890
    s4 = 0.5073767736746132
    B1 = 0.0716134247098111
    B2 = 0.4540903525515453
    B3 = 0.0427846154667780
    B4 = 0.2157558036359238
    
    x_nodes = np.array([ r1,  r2,  r4,  r3,  s4,   0,   0, -s4, -r3, -r4,
                        -r1, -r2, -r4, -r3, -s4,   0,   0,  s4,  r3,  r4])
    y_nodes = np.array([  0,   0,  s4,  r3,  r4,  r1,  r2,  r4,  r3,  s4,
                          0,   0, -s4, -r3, -r4, -r1, -r2, -r4, -r3, -s4])
    weights = np.array([ B1,  B2,  B4,  B3,  B4,  B1,  B2,  B4,  B3,  B4,
                         B1,  B2,  B4,  B3,  B4,  B1,  B2,  B4,  B3,  B4])
    
    return (x_nodes, y_nodes), weights

def chanut_21pt():
    '''
    Chanut's first 21-point rule (Stroud C2: 9-2):
    
    A ninth-order rule with 21 points.
    
    Chanut, A., "Calcul numérique des integrales doubles",
    C. R. Acad. Sci. Paris, v. 256, 1963, pp. 3239-3241.
    '''
    r = 0.9490600754
    s = 0.7012653741
    t = 0.4889268570
    u = 0.8539562957
    v = 0.07689419029
    B0 = 0.5264789704
    B1 = 0.0988411238
    B2 = 0.1383642832
    B3 = 0.3943019434
    
    x_nodes = np.array([ 0,  u,  r,  t,  s,  v, -v, -s, -t, -r, -u,
                            -u, -r, -t, -s, -v,  v,  s,  t,  r,  u])
    y_nodes = np.array([ 0,  v,  s,  t,  r,  u,  u,  r,  t,  s,  v,
                            -v, -s, -t, -r, -u, -u, -r, -t, -s, -v])
    weights = np.array([B0, B2, B1, B3, B1, B2, B2, B1, B3, B1, B2,
                            B2, B1, B3, B1, B2, B2, B1, B3, B1, B2])
    
    return (x_nodes, y_nodes), weights

def chanut_21pt2():
    '''
    Chanut's second 21-point rule (Stroud C2: 9-2):
    
    A ninth-order rule with 21 points.
    
    Chanut, A., "Calcul numérique des integrales doubles",
    C. R. Acad. Sci. Paris, v. 256, 1963, pp. 3239-3241.
    '''
    r = 0.9648021210
    s = 0.3480000007
    t = 0.8536348540
    u = 0.6564038459
    v = 0.3017236686
    B0 = 0.5267489720
    B1 = 0.08574340975
    B2 = 0.2828874118
    B3 = 0.1310511139
    
    x_nodes = np.array([ 0,  u,  r,  t,  s,  v, -v, -s, -t, -r, -u,
                            -u, -r, -t, -s, -v,  v,  s,  t,  r,  u])
    y_nodes = np.array([ 0,  v,  s,  t,  r,  u,  u,  r,  t,  s,  v,
                            -v, -s, -t, -r, -u, -u, -r, -t, -s, -v])
    weights = np.array([B0, B2, B1, B3, B1, B2, B2, B1, B3, B1, B2,
                            B2, B1, B3, B1, B2, B2, B1, B3, B1, B2])
    
    return (x_nodes, y_nodes), weights

def chanut_25pt():
    '''
    Chanut's first 25-point rule (Stroud C2: 9-3):
    
    A ninth-order rule with 25 points.
    
    Chanut, A., "Calcul numérique des integrales doubles",
    C. R. Acad. Sci. Paris, v. 256, 1963, pp. 3239-3241.
    '''
    r = 0.9446995561
    s = 0.7128121207
    t = 0.5992626637
    u = 0.8880649884
    v = 0.1204163075
    w = 0.3703646943
    B0 = 0.5267489673
    B1 = 0.09443701609
    B2 = 0.1058119207
    B3 = 0.2339074423
    
    x_nodes = array([ 0,  r,  u,  t,  w,  v,  s, -s, -v, -w, -t, -u, -r,
                         -r, -u, -t, -w, -v, -s,  s,  v,  w,  t,  u,  r])
    y_nodes = array([ 0,  s,  v,  w,  t,  u,  r,  r,  u,  t,  w,  v,  s,
                         -s, -v, -w, -t, -u, -r, -r, -u, -t, -w, -v, -s])
    weights = array([B0, B1, B2, B3, B3, B2, B1, B1, B2, B3, B3, B2, B1,
                         B1, B2, B3, B3, B2, B1, B1, B2, B3, B3, B2, B1])
    
    return (x_nodes, y_nodes), weights

def chanut_25pt2():
    '''
    Chanut's second 25-point rule (Stroud C2: 9-3):
    
    A ninth-order rule with 25 points.
    
    Chanut, A., "Calcul numérique des integrales doubles",
    C. R. Acad. Sci. Paris, v. 256, 1963, pp. 3239-3241.
    '''
    r = 0.9477024551
    s = 0.7049610584
    t = 0.8664350908
    u = 0.5607932266
    v = 0.4136613878
    w = 0.08752771858
    B0 = 0.5267489717
    B1 = 0.09728367154
    B2 = 0.2121324628
    B3 = 0.1247402442
    
    x_nodes = array([ 0,  r,  u,  t,  w,  v,  s, -s, -v, -w, -t, -u, -r,
                         -r, -u, -t, -w, -v, -s,  s,  v,  w,  t,  u,  r])
    y_nodes = array([ 0,  s,  v,  w,  t,  u,  r,  r,  u,  t,  w,  v,  s,
                         -s, -v, -w, -t, -u, -r, -r, -u, -t, -w, -v, -s])
    weights = array([B0, B1, B2, B3, B3, B2, B1, B1, B2, B3, B3, B2, B1,
                         B1, B2, B3, B3, B2, B1, B1, B2, B3, B3, B2, B1])
    
    return (x_nodes, y_nodes), weights

def rr_25pt():
    '''
    Rabinowitz-Richter 25-point rule (Stroud C2: 11-1):
    
    An eleventh-order rule with 25 points. Four points are outside the square.
    
    Rabinowitz, P., and Richter, N., "Perfectly symmetric two-dimensional
    integration formulas with minimal numbers of points", Math. Comput.,
    v. 23, 1969, pp. 765-779.
    '''
    r1 = 0.7697990683966493
    r2 = 1.044402915409813
    r3 = 0.4134919534491139
    r4 = 0.9357870124405403
    r5 = 0.5756535958404649
    s5 = 0.8830255085256902
    A = 0.3653795255859022
    B1 = 0.2442720577517539
    B2 = 0.0277561655642043
    B3 = 0.3089930361337136
    B4 = 0.0342651038512293
    B5 = 0.1466843776513117
    
    x_nodes = np.array([ 0,  r1,  r2,  s5,  r3,  r4,  r5,
                              0,   0, -r5, -r3, -r4, -s5,
                            -r1, -r2, -s5, -r3, -r4, -r5
                              0,   0,  r5,  r3,  r4,  s5])
    y_nodes = np.array([ 0,   0,   0,  r5,  r3,  r4,  s5,
                             r1,  r2,  s5,  r3,  r4,  r5,
                              0,   0, -r5, -r3, -r4, -s5,
                            -r1, -r2, -s5, -r3, -r4, -r5])
    weights = np.array([ A,  B1,  B2,  B5,  B3,  B4,  B5,
                             B1,  B2,  B5,  B3,  B4,  B5,
                             B1,  B2,  B5,  B3,  B4,  B5,
                             B1,  B2,  B5,  B3,  B4,  B5])
    
    return (x_nodes, y_nodes), weights

def rr_28pt():
    '''
    Rabinowitz-Richter 28-point rule (Stroud C2: 11-2):
    
    An eleventh-order rule with 28 points.
    
    Rabinowitz, P., and Richter, N., "Perfectly symmetric two-dimensional
    integration formulas with minimal numbers of points", Math. Comput.,
    v. 23, 1969, pp. 765-779.
    '''
    r1 = 0.8989737240828844
    r2 = 0.7632367891419969
    r3 = 0.8949648832822285
    r4 = 0.6322452037101431
    r5 = 0.2797353125538562
    r6 = 0.9602661668053869
    s6 = 0.4347413023856830
    B1 = 0.0176679598882646
    B2 = 0.2322248008989674
    B3 = 0.0715516745178401
    B4 = 0.2192868905662522
    B5 = 0.2965842326220580
    B6 = 0.0813422207533089
    
    x_nodes = np.array([ r1,  r2,  r6,  r3,  r4,  r5,  s6,
                          0,   0, -s6, -r3, -r4, -r5, -r6,
                        -r1, -r2, -r6, -r3, -r4, -r5, -r6,
                          0,   0,  s6,  r3,  r4,  r5,  s6])
    y_nodes = np.array([  0,   0,  s6,  r3,  r4,  r5,  r6,
                         r1,  r2,  r6,  r3,  r4,  r5,  r6,
                          0,   0, -s6, -r3, -r4, -r5, -r6,
                        -r1, -r2, -r6, -r3, -r4, -r5, -r6])
    weights = np.array([ B1,  B2,  B6,  B3,  B4,  B5,  B6,
                         B1,  B2,  B6,  B3,  B4,  B5,  B6,
                         B1,  B2,  B6,  B3,  B4,  B5,  B6,
                         B1,  B2,  B6,  B3,  B4,  B5,  B6])
    
    return (x_nodes, y_nodes, weights)

def rr_37pt():
    '''
    Rabinowitz-Richter 37-point rule (Stroud C2: 13-1):
    
    A thirteenth-order rule with 37 points.
    
    Rabinowitz, P., and Richter, N., "Perfectly symmetric two-dimensional
    integration formulas with minimal numbers of points", Math. Comput.,
    v. 23, 1969, pp. 765-779.
    '''
    r1 = 0.9909890363004588
    r2 = 0.6283940712305196
    r3 = 0.9194861553393097
    r4 = 0.6973201917871096
    r5 = 0.3805687186904865
    r6 = 0.9708504361720127
    s6 = 0.6390348393207334
    r7 = 0.8623637916722844
    s7 = 0.3162277660168378
    A = 0.2995235559387052
    B1 = 0.0331100668669073
    B2 = 0.1802214941550577
    B3 = 0.0391672789603492
    B4 = 0.1387748348777338
    B5 = 0.2268881207335663
    B6 = 0.0365739576550950
    B7 = 0.1169047000557597
    
    x_nodes = np.array([0,  r1,  r2,  r7,  r6,  r3,  r4,  r5,  s6,  s7,
                             0,   0, -s7, -s6, -r3, -r4, -r5, -r6, -r7,
                           -r1, -r2, -r7, -r6, -r3, -r4, -r5, -s6, -s7,
                             0,   0,  s7,  s6,  r3,  r4,  r5,  r6,  r7])
    y_nodes = np.array([0,   0,   0,  s7,  s6,  r3,  r4,  r5,  r6,  r7,
                            r1,  r2,  r7,  r6,  r3,  r4,  r5,  s6,  s7,
                             0,   0, -s7, -s6, -r3, -r4, -r5, -r6, -r7,
                           -r1, -r2, -r7, -r6, -r3, -r4, -r5, -s6, -s7])
    weights = np.array([A,  B1,  B2,  B7,  B6,  B3,  B4,  B5,  B6,  B7,
                            B1,  B2,  B7,  B6,  B3,  B4,  B5,  B6,  B7,
                            B1,  B2,  B7,  B6,  B3,  B4,  B5,  B6,  B7,
                            B1,  B2,  B7,  B6,  B3,  B4,  B5,  B6,  B7])
    
    return (x_nodes, y_nodes), weights
    

def rr_44pt():
    '''
    Rabinowitz-Richter 44-point rule (Stroud C2: 15-1):
    
    A fifteenth-order rule with 44 points. Four points are outside the square.
    
    Rabinowitz, P., and Richter, N., "Perfectly symmetric two-dimensional
    integration formulas with minimal numbers of points", Math. Comput.,
    v. 23, 1969, pp. 765-779.
    '''
    r1 = 1.315797935069747
    r2 = 0.9796158388578564
    r3 = 0.6375456844500517
    r4 = 0.9346799288936658
    r5 = 0.7662665721615083
    r6 = 0.5138362475917853
    r7 = 0.2211895845055072
    r8 = 0.9769495664551867
    s8 = 0.6375975639376926
    r9 = 0.8607803779721935
    s9 = 0.3368688874716777
    B1 = -4.0980941939297e-6
    B2 = 0.0414134647558384
    B3 = 0.1837583771750436
    B4 = 0.0280217865486269
    B5 = 0.0948146979601645
    B6 = 0.1688054053337613
    B7 = 0.1898474000367674
    B8 = 0.0331477474104121
    B9 = 0.1135237357315838
    
    x_nodes = np.array([ r1,  r2,  r3,  r4,  r5,  r6,  r7,  r8,  s8,  r9,  s9,
                          0,   0,   0, -r4, -r5, -r6, -r7, -s8, -r8, -s9, -r9,
                        -r1, -r2, -r3, -r4, -r5, -r6, -r7, -r8, -s8, -r9, -s9,
                          0,   0,   0,  r4,  r5,  r6,  r7,  s8,  r8,  s9,  r9])
    y_nodes = np.array([  0,   0,   0,  r4,  r5,  r6,  r7,  s8,  r8,  s9,  r9,
                         r1,  r2,  r3,  r4,  r5,  r6,  r7,  r8,  s8,  r9,  s9,
                          0,   0,   0, -r4, -r5, -r6, -r7, -s8, -r8, -s9, -r9,
                        -r1, -r2, -r3, -r4, -r5, -r6, -r7, -r8, -s8, -r9, -s9])
    weights = np.array([ B1,  B2,  B3,  B4,  B5,  B6,  B7,  B8,  B8,  B9,  B9,
                         B1,  B2,  B3,  B4,  B5,  B6,  B7,  B8,  B8,  B9,  B9,
                         B1,  B2,  B3,  B4,  B5,  B6,  B7,  B8,  B8,  B9,  B9,
                         B1,  B2,  B3,  B4,  B5,  B6,  B7,  B8,  B8,  B9,  B9])
    
    return (x_nodes, y_nodes), weights

def rr_48pt():
    '''
    Rabinowitz-Richter 48-point rule (Stroud C2: 15-2):
    
    A fifteenth-order rule with 48 points.
    
    Rabinowitz, P., and Richter, N., "Perfectly symmetric two-dimensional
    integration formulas with minimal numbers of points", Math. Comput.,
    v. 23, 1969, pp. 765-779.
    '''
    r1 = 0.9915377816777667
    r2 = 0.8020163879230440
    r3 = 0.5648674875232742
    r4 = 0.9354392392539896
    r5 = 0.7624563338825799
    r6 = 0.2156164241427213
    r7 = 0.9769662659711761
    s7 = 0.6684480048977932
    r8 = 0.8937128379503403
    s8 = 0.3735205277617582
    r9 = 0.6122485619312083
    s9 = 0.4078983303613935
    B1 = 0.0301245207981210
    B2 = 0.0871146840209092
    B3 = 0.1250080294351494
    B4 = 0.0267651407861666
    B5 = 0.0959651863624437
    B6 = 0.1750832998343375
    B7 = 0.0283136372033274
    B8 = 0.0866414716025093
    B9 = 0.1150144605755996
    
    x_nodes = np.array([ r1,  r2,  r3,  r4,  r5,  r6,  r7,  s7,  r8,  s8,  r9,  s9,
                          0,   0,   0, -r4, -r5, -r6, -s7, -r7, -s8, -r8, -s9, -r9,
                        -r2, -r2, -r3, -r4, -r5, -r6, -r7, -s8, -r8, -s8, -r9, -s9,
                          0,   0,   0,  r4,  r5,  r6,  s7,  r7,  s8,  r8,  s9,  r9])
    y_nodes = np.array([  0,   0,   0,  r4,  r5,  r6,  s7,  r7,  s8,  r8,  s9,  r9,
                         r1,  r2,  r3,  r4,  r5,  r6,  r7,  s7,  r8,  s8,  r9,  s9,
                          0,   0,   0, -r4, -r5, -r6, -s7, -r7, -s8, -r8, -s9, -r9,
                        -r1, -r2, -r3, -r4, -r5, -r6, -r7, -s7, -r8, -s8, -r9, -s9])
    weights = np.array([ B1,  B2,  B3,  B4,  B5,  B6,  B7,  B7,  B8,  B8,  B9,  B9,
                         B1,  B2,  B3,  B4,  B5,  B6,  B7,  B7,  B8,  B8,  B9,  B9,
                         B1,  B2,  B3,  B4,  B5,  B6,  B7,  B7,  B8,  B8,  B9,  B9,
                         B1,  B2,  B3,  B4,  B5,  B6,  B7,  B7,  B8,  B8,  B9,  B9])
    
    return (x_nodes, y_nodes), weights

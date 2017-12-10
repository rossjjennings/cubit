import numpy as np
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
    pass

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
    pass

def ac_7pt():
    '''
    Albrecht-Collatz seven-point rule (Stroud C2: 5-2):
    
    A fifth-order fule with seven points. The points are distributed
    symmetrically with respect to the diagonals.
    
    Albrecht, J. and Collatz, L., "Zur numerischen auswertung mehrdimensionaler
    Integrale", Z. Agnew. Math. Mech., v. 38, 1958, pp. 1-15.
    '''
    pass

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
    pass

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
    pass

def meister_13pt():
    '''
    Meister's thirteen-point rule (Stroud C2: 5-6):
    
    A fifth-order rule with thirteen points. Four points lie at corners,
    and four lie along edges.
    
    Meister, Bernd, "On a family of cubature formulae",
    Computer J., v. 8, 1966, pp. 368-371.
    '''
    pass

def irwin_24pt():
    '''
    Irwin's 24-point rule (Stroud C2: 5-7):
    
    A fifth-order rule with 24 points. Eight points have negative weights.
    Four lie at the corners, and the rest are outside the region.
    
    Irwin, J. O., "On quadrature and cubature",
    Tracts for Computers, No. 10, 1923.
    '''
    pass

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
    pass

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
    pass

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
    pass

def tyler_12pt2():
    '''
    Tyler's second twelve-point rule (Stroud C2: 7-5):
    
    A seventh-order rule with twelve points. Four points have negative weight.
    
    Tyler, G. W., "Numerical integration of functions of several variables",
    Canad. J. Math., v. 5, 1953, pp. 393-412.
    '''
    pass

def meister_25pt():
    '''
    Meister's 25-point rule (Stroud C2: 7-6):
    
    A seventh-order rule with 25 points. Four points have negative weight.
    
    Meister, Bernd, "On a family of cubature formulae",
    Computer J., v. 8, 1966, pp. 368-371.
    '''
    pass

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

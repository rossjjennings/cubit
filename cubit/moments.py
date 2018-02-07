import sympy as sym

x = sym.Symbol('x')
class weight_fn:
    def __init__(self, momfunc):
        self.moment = momfunc
    
    def inner(p1, p2):
        prod = sym.Poly(p1*p2, x)
        iter = enumerate(reversed(prod.all_coeffs()))
        return sum(coeff*self.moment(i) for i, coeff in iter)
    
    def sqnorm(p):
        return self.inner(p, p)
    
    def orthopoly(self, n, cache = {}):
        try:
            return cache[n]
        except KeyError:
            if n == 0:
                cache[n] = sym.Integer(1)
            elif n == 1:
                cache[n] = x - self.moment(1)/moment(0)
            else:
                twoprev = self.orthopoly(n - 2)
                oneprev = self.orthopoly(n - 1)
                twoprev_sqnorm = self.sqnorm(twoprev)
                oneprev_sqnorm = self.sqnorm(oneprev)
                product = self.inner(oneprev, x*oneprev)
                alpha = product/oneprev_sqnorm
                beta = oneprev_sqnorm/twoprev_sqnorm
                poly = (x - alpha)*oneprev - beta*twoprev
                cache[n] = sym.expand(poly)
            return cache[n]
    return momfunc_orthopoly

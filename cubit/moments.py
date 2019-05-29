import sympy as sym

x = sym.Symbol('x')

class weight_fn:
    def __init__(self, momfunc):
        self.moment = momfunc
        self._cache = dict()
    
    def inner(self, p1, p2):
        prod = sym.Poly(p1*p2, x)
        iter = enumerate(reversed(prod.all_coeffs()))
        return sum(coeff*self.moment(i) for i, coeff in iter)
    
    def sqnorm(self, p):
        return self.inner(p, p)
    
    def orthopoly(self, n):
        try:
            return self._cache[n]
        except KeyError:
            if n == 0:
                self._cache[n] = sym.Integer(1)
            elif n == 1:
                self._cache[n] = x - self.moment(1)/self.moment(0)
            else:
                twoprev = self.orthopoly(n - 2)
                oneprev = self.orthopoly(n - 1)
                twoprev_sqnorm = self.sqnorm(twoprev)
                oneprev_sqnorm = self.sqnorm(oneprev)
                product = self.inner(oneprev, x*oneprev)
                alpha = product/oneprev_sqnorm
                beta = oneprev_sqnorm/twoprev_sqnorm
                poly = (x - alpha)*oneprev - beta*twoprev
                self._cache[n] = sym.expand(poly)
            return self._cache[n]

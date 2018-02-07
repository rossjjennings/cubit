import sympy as sym

def inner(p1, p2, momfunc):
    prod = sym.Poly(p1*p2, x)
    iter = enumerate(reversed(prod.all_coeffs()))
    return sum(coeff*momfunc(i) for i, coeff in iter)

def sqnorm(p, momfunc):
    prod = sym.Poly(p**2, x)
    iter = enumerate(reversed(prod.all_coeffs()))
    return sum(coeff*momfunc(i) for i, coeff in iter)

def orthopoly(momfunc):
    def momfunc_orthopoly(n, cache = {}):
        try:
            return cache[n]
        except KeyError:
            if n == 0:
                cache[n] = sym.Integer(1)
            elif n == 1:
                cache[n] = x - momfunc(1)/momfunc(0)
            else:
                twoprev = momfunc_orthopoly(n - 2)
                oneprev = momfunc_orthopoly(n - 1)
                twoprev_sqnorm = sqnorm(twoprev, momfunc)
                oneprev_sqnorm = sqnorm(oneprev, momfunc)
                product = inner(oneprev, x*oneprev, momfunc)
                alpha = product/oneprev_sqnorm
                beta = oneprev_sqnorm/twoprev_sqnorm
                poly = (x - alpha)*oneprev - beta*twoprev
                cache[n] = sym.expand(poly)
            return cache[n]
    return momfunc_orthopoly

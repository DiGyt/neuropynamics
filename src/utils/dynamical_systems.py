import numpy as np
from sympy import symbols, solve, lambdify, sympify, dsolve, Eq, solveset, linear_eq_to_matrix, nonlinsolve, Matrix, diff, sqrt
from scipy import integrate

def rk4( f, x0, y0, h, xn ):
    n = int((xn - x0)/h)
    x = np.array( [ x0 ] * n, dtype=float )
    y = np.array( [ y0 ] * n, dtype=float )
    for i in range( n-1 ):
        k1 = h * f( x[i], y[i] )
        k2 = h * f( x[i] + 0.5 * h, y[i] + 0.5 * k1 )
        k3 = h * f( x[i] + 0.5 * h, y[i] + 0.5 * k2 )
        k4 = h * f( x[i] + h, y[i] + k3 )
        
        x[i+1] = x[i] + h
        y[i+1] = y[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0
        
    return y, x

def plotEquation(eq, solvar, plotvar, inputrange):
    '''
        Given a sympy equation object `eq`, this function return a series of values,
        in the range of `inputrange` such that the equation is solved for
        `solvar` and the input is for `plotvar`
    '''
    eq = Eq(eq, 0)
    sol = solve(eq, solvar)
    vals = []
    for s in sol:
        f = lambdify((plotvar), sol)
        vals.append([f(i) for i in inputrange])

    return vals

def findFixedPoints(f, g, xrange, yrange):
    return [(x, y) for x in xrange for y in yrange if f(x, y) == 0 and g(x, y) == 0]


def system(initialx, initialy, func1, func2, delta, timerange):
    x = [initialx]
    y = [initialy]
    
    for i in range(timerange):
        x.append(x[i] + func1(x[i], y[i]) * delta)
        y.append(y[i] + func2(x[i], y[i]) * delta)
    
    return x, y
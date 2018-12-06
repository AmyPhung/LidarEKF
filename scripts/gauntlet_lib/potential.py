import numpy as np
from sympy.abc import t, m, b, x, y, u, v
from sympy.physics.vector import ReferenceFrame, gradient
from sympy import integrate, re, lambdify, sqrt

# Potential field stuff
R = ReferenceFrame('R')

h = m * t + b
g = (m ** 2 + 1) ** .5 / ((t - R[0]) ** 2 + (h - R[1]) ** 2)
repulsemat = gradient(integrate(g, (t, u, v)), R).to_matrix(R)
repulsex = lambdify((m, b, u, v, R[0], R[1]), re(repulsemat[0]))
repulsey = lambdify((m, b, u, v, R[0], R[1]), re(repulsemat[1]))


attrmat = gradient((x - R[0]) ** 2 + (y - R[1]) ** 2, R).to_matrix(R)
attrx = lambdify((x, y, R[0], R[1]), re(attrmat[0]))
attry = lambdify((x, y, R[0], R[1]), re(attrmat[1]))


def repulse(m, b, u, v, r0, r1):
    '''Get the gradient of the repulsion field'''
    return np.array([repulsex(m, b, u, v, r0, r1), repulsey(m, b, u, v, r0, r1)])


def attr(x, y, r0, r1):
    '''Get the gradient of the attraction field'''
    return np.array([attrx(x, y, r0, r1), attry(x, y, r0, r1)])

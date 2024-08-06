# Calculation of Trigonometric Fourier Series of piecewise functions

import math
import numpy as np


def gauss_quad_str(f, a, b, w0):
    #  Gauss-Legendre quadrature
    #    f: String function (with time 't' and fundamental frequency 'w0')
    #    a, b: Interval
    #    w0: Fundamental frequency
    points = np.array([-0.949107912342758, -0.741531185599394, -0.405845151377397,
                       0, 0.405845151377397, 0.741531185599394, 0.949107912342758])
    weights = np.array([0.129484966168870, 0.279705391489277, 0.381830050505119,
                        0.417959183673469, 0.381830050505119, 0.279705391489277, 0.129484966168870])
    z = 0
    for i in range(len(points)):
        t = ((b-a)*points[i]+(b+a))/2
        z += weights[i]*eval(f)
    z *= (b - a)/2
    return z


def automatic_gauss_quad_str(f, a, b, w0, maxerr):
    z_old = gauss_quad_str(f, a, b, w0)
    mid = (a + b)/2
    z1 = gauss_quad_str(f, a, mid, w0)
    z2 = gauss_quad_str(f, mid, b, w0)
    z = z1 + z2
    # Recursive part
    error = 1 - z/z_old
    if abs(error) > maxerr:
        z = automatic_gauss_quad_str(f, a, mid, w0, maxerr) + automatic_gauss_quad_str(f, mid, b, w0, maxerr)

    return z


def calculate_coefficients(piecewise_f, init_t, T, n_terms, maxerr):
    # a, b = calculate_coefficients
    #       piecewise_f: Dictionary with functions and upper limit
    #       init_t: t0
    #       T: Period
    w0 = 2*math.pi/(T - init_t)
    a = np.array(np.zeros([1, n_terms]))
    b = np.array(np.zeros([1, n_terms]))
    for n in range(n_terms):
        inf_lim = init_t
        for f, t in piecewise_f.items():
            f_cos = f + '*math.cos(' + str(n) + '*w0*t)'
            f_sin = f + '*math.sin(' + str(n) + '*w0*t)'
            a[0][n] += 2*automatic_gauss_quad_str(f_cos, inf_lim, t, w0, maxerr)/(T - init_t)
            b[0][n] += 2*automatic_gauss_quad_str(f_sin, inf_lim, t, w0, maxerr)/(T - init_t)
            inf_lim = t
    a[0][0] /= 2
    print(a)
    print(b)
    return a, b


def evaluate_piecewise_function(piecewise_f, t0, T, n_points):
    # Evaluate function
    # a, b = evaluate_piecewise_function
    #       piecewise_f: Dictionary with functions and upper limit
    #       t0: t0
    #       T: Period
    #       n_points: Points to calculate
    inf_lim = t0
    f_orig = np.array([])
    t_ls = np.linspace(t0, T, n_points)
    for f, t_ in piecewise_f.items():
        t = t_ls[t_ls >= inf_lim]
        if t_ != T:
            t = t[t < t_]
        else:
            t = t[t <= t_]
        if 't' not in f:
            # f is scalar, need to generate vector
            fe = eval(f)*np.ones([1, t.size])
            f_orig = np.concatenate((f_orig, fe[0]))
        else:
            fe = eval(f)
            f_orig = np.concatenate((f_orig, eval(f)))
        inf_lim = t_
    return t_ls, f_orig


def calculate_fourier_approximation(a, b, t_ls):
    init_t = 0
    f_hat = a[0][0]
    w0 = 2 * math.pi / (max(t_ls) - min(t_ls))
    for n in range(1, n_terms):
        f_hat = np.add(f_hat, a[0][n] * np.cos(n * w0 * t_ls))
        f_hat = np.add(f_hat, b[0][n] * np.sin(n * w0 * t_ls))
    return f_hat


# Main program
piecewise_f = {'t**2': 1, '0': T}
n_points = 128
maxerr = 1e-9
a, b = calculate_coefficients(piecewise_f, t0, T, n_terms, maxerr)
t_ls, f_orig = evaluate_piecewise_function(piecewise_f, t0, T, n_points)
f_hat = calculate_fourier_approximation(a, b, t_ls)

import matplotlib.pyplot as plt
plt.plot(t_ls, f_orig, t_ls, f_hat)
plt.show()

"""
Polynomial root finder with improved precision
Mauricio Martinez-Garcia, 2007. Revised: 2014, 2016, 2024
"""
import numpy as np

def polygcf(p, q, tol):
    #   Uses the Euclid's algorithm to find the greatest common factor
    #   of polynomials with coefficients 'p' and 'q', with residue tolerance 'tol'
    n2 = np.linalg.norm(p)
    [a, r0] = np.polydiv(p, q)
    if np.linalg.norm(r0) > n2*tol:
        [a, r1] = np.polydiv(q, r0)
        if np.linalg.norm(r1) > n2*tol:
            rn = 1
            while np.linalg.norm(rn) > n2*tol:
                [a, rn] = np.polydiv(r0, r1)
                r0 = r1
                r1 = rn
            g = r0
        else:
            g = r0
    else:
        g = q
    return g

def rootmult(p, r, tol):
    #   Finds the multiplicity of root 'r' in polynomial with coefficients 'p'
    #   with residue tolerance 'tol'
    n2 = np.linalg.norm(p)
    v = np.abs(np.polyval(p, r))
    n = 0
    if v > n2*tol:
        print('Error: Value is not root')
    else:
        while v < n2*tol:
            p = np.polyder(p)
            v = np.abs(np.polyval(p, r))
            n += 1
    return n


def mroots(p, tol=1e-8):
    # First, divide the input polynomial by the
    # greatest common factor of itself and its first derivative
    q = np.polyder(p)
    d = polygcf(p, q, tol)
    pr = np.polydiv(p, d)[0]
    # Second, find roots of resulting polynomial,
    # with (hopefully) single roots
    rr = np.roots(pr)
    # Finally, determine multiplicity of each root
    r = np.array([])
    for i in range(np.size(rr)):
        r = np.append(r, np.multiply(rr[i], np.ones([1, rootmult(p, rr[i], tol)]))[0])
    return r


# How removing root multiplicity can improve the accuracy of
# Numpy function 'roots'.

# First, define a polynomial with multiple roots.
p = np.polymul(np.array([1, 4, 6, 4, 1]), np.array([1, 2, -8]))

# Now, use 'roots' to compute the roots.
r1 = np.roots(p)
print(r1)

# Finally, use 'mroots' for improved accuracy.
r2 = mroots(p)
print(r2)

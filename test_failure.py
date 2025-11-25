from math import exp, sqrt, erf, ceil
from math import factorial as fac

def gaussian_center_weight(sigma, t):
    """ Weight of the gaussian of std deviation s, on the interval [-t, t]
    :param x: (float)
    :param y: (float)
    :returns: erf( t / (sigma*\sqrt 2) )
    """
    return erf(t / (sigma * sqrt(2.)))


def binomial(x, y):
    """ Binomial coefficient
    :param x: (integer)
    :param y: (integer)
    :returns: y choose x
    """
    try:
        binom = fac(x) // fac(y) // fac(x - y)
    except ValueError:
        binom = 0
    return binom


def centered_binomial_pdf(k, x):
    """ Probability density function of the centered binomial law of param k at x
    :param k: (integer)
    :param x: (integer)
    :returns: p_k(x)
    """
    return binomial(2*k, x+k) / 2.**(2*k)


def build_centered_binomial_law(k):
    """ Construct the binomial law as a dictionnary
    :param k: (integer)
    :param x: (integer)
    :returns: A dictionnary {x:p_k(x) for x in {-k..k}}
    """
    D = {}
    for i in range(-k, k+1):
        D[i] = centered_binomial_pdf(k, i)
    return D

def build_centered_normal(sigma):
    """ Construct the normal law as a dictionary
    :param k: (integer)
    :param x: (integer)
    :returns: A dictionnary {x:p_k(x) for x in {-k..k}}
    """

    kmax = 1
    while True:
        partial_weight = 2*exp(-kmax**2/(2*sigma**2))
        if partial_weight < 2**(-400):
            break
        kmax += 1
    
    weight = 1
    for k in range(kmax, 0, -1):
        partial_weight = 2*exp(-k**2/(2*sigma**2))
        weight += partial_weight
        k += 1

    D = {}
    for i in range(-kmax, kmax+1):
        D[i] = exp(-i**2/(2*sigma**2))/weight

    return D


def mod_switch(x, q, rq):
    """ Modulus switching (rounding to a different discretization of the Torus)
    :param x: value to round (integer)
    :param q: input modulus (integer)
    :param rq: output modulus (integer)
    """
    return int(round(1.* rq * x / q) % rq)


def mod_centered(x, q):
    """ reduction mod q, centered (ie represented in -q/2 .. q/2)
    :param x: value to round (integer)
    :param q: input modulus (integer)
    """
    a = x % q
    if a < q/2:
        return a
    return a - q


def build_mod_switching_error_law(q, rq):
    """ Construct Error law: law of the difference introduced by switching from and back a uniform value mod q
    :param q: original modulus (integer)
    :param rq: intermediate modulus (integer)
    """
    D = {}
    V = {}
    for x in range(q):
        y = mod_switch(x, q, rq)
        z = mod_switch(y, rq, q)
        d = mod_centered(x - z, q)
        D[d] = D.get(d, 0) + 1./q
        V[y] = V.get(y, 0) + 1

    return D


def law_convolution(A, B):
    """ Construct the convolution of two laws (sum of independent variables from two input laws)
    :param A: first input law (dictionnary)
    :param B: second input law (dictionnary)
    """

    C = {}
    for a in A:
        for b in B:
            c = a+b
            C[c] = C.get(c, 0) + A[a] * B[b]
    return C


def law_product(A, B):
    """ Construct the law of the product of independent variables from two input laws
    :param A: first input law (dictionnary)
    :param B: second input law (dictionnary)
    """
    C = {}
    for a in A:
        for b in B:
            c = a*b
            C[c] = C.get(c, 0) + A[a] * B[b]
    return C

def clean_dist(A):
    """ Clean a distribution to accelerate further computation (drop element of the support with proba less than 2^-300)
    :param A: input law (dictionnary)
    """
    B = {}
    for (x, y) in A.items():
        if y>2**(-400):
            B[x] = y
    return B


def iter_law_convolution(A, i):
    """ compute the -ith forld convolution of a distribution (using double-and-add)
    :param A: first input law (dictionnary)
    :param i: (integer)
    """
    D = {0: 1.0}
    i_bin = bin(i)[2:]  # binary representation of n
    for ch in i_bin:
        print("iter", ch)
        D = law_convolution(D, D)
        D = clean_dist(D)
        if ch == '1':
            D = law_convolution(D, A)
            D = clean_dist(D)

    return D


def tail_probability(D, t):
    '''
    Probability that an drawn from D is strictly greater than t in absolute value
    :param D: Law (Dictionnary)
    :param t: tail parameter (integer)
    '''
    s = 0
    ma = max(D.keys())
    if t >= ma:
        return 0
    for i in reversed(range(int(ceil(t)), ma)):  # Summing in reverse for better numerical precision (assuming tails are decreasing)
        s += D.get(i, 0) + D.get(-i, 0)
    return s


def find_tail_for_probability(D, p):
    s = 0
    ma = max(D.keys())
    for i in reversed(range(1, ma)):  # Summing in reverse for better numerical precision (assuming tails are decreasing)
        s += D.get(i, 0) + D.get(-i, 0)
        if s >= p:
            return i-1
    return s

import math
from math import log

bitsec = 128
(n, k) = (512, 1) 

def build_centered_binomial(param):
    D = {}
    for i in range(-param, param+1):
        if abs(i) <= param:
            binom_coeff = math.comb(2*param, param + i)
            prob = binom_coeff / (4 ** param)
            if prob > 1e-12:
                D[i] = prob
    return D

def multiply_distribution(D, factor):
    """Multiply factor"""
    new_D = {}
    for x, prob in D.items():
        new_x = factor * x
        new_D[new_x] = new_D.get(new_x, 0) + prob
    return new_D


CBD3 = build_centered_binomial(4)  # S_j, S_i, E_i 
CBD2 = build_centered_binomial(4)  # E_j

print("Basic Distribution...")

# S_j = sj0 + sj1 
S_j = CBD3
for _ in range(1):
    S_j = law_convolution(S_j, CBD3)
    S_j = clean_dist(S_j)

# E_i = ei0 + ei1 + ei2
E_i = CBD3
for _ in range(2):
    E_i = law_convolution(E_i, CBD3)
    E_i = clean_dist(E_i)

# S_i = si0 + si1 + si2
S_i = CBD3
for _ in range(2):
    S_i = law_convolution(S_i, CBD3)
    S_i = clean_dist(S_i)

# E_j = ej0 + ej1 
E_j = CBD2
for _ in range(1):
    E_j = law_convolution(E_j, CBD2)
    E_j = clean_dist(E_j)


# 2S_jᵀE_i
S_jT_E_i = law_product(S_j, E_i)
S_jT_E_i = clean_dist(S_jT_E_i)
S_jT_E_i_x2 = multiply_distribution(S_jT_E_i, 2)  # 2S_jᵀE_i

# 2E_jᵀS_i
E_jT_S_i = law_product(E_j, S_i)
E_jT_S_i = clean_dist(E_jT_S_i)
E_jT_S_i_x2 = multiply_distribution(E_jT_S_i, 2)  # 2E_jᵀS_i


# target_prob = 1e-85 / min(n, 256)
target_prob = 2**(-155) / min(n, 256)

print("Final Noise...")

# n*k of single 2si*ej
S_jT_E_i_sum = iter_law_convolution(S_jT_E_i_x2, n * k)
E_jT_S_i_sum = iter_law_convolution(E_jT_S_i_x2, n * k)

# final 2S_jᵀE_i - 2E_jᵀS_i
diff = law_convolution(S_jT_E_i_sum, {-x: p for x, p in E_jT_S_i_sum.items()})
diff = clean_dist(diff)

# 2y
E=multiply_distribution(CBD2, 2)
E = clean_dist(E)
En=iter_law_convolution(E, n)
En = clean_dist(En)
# final 2S_jᵀE_i - 2E_jᵀS_i+2y
diff = law_convolution(diff, En)
diff = clean_dist(diff)

print("Calculate final...")

# tail-bound
f_final = find_tail_for_probability(diff, target_prob)
print(f"Final tail bound (2S_jᵀE_i - 2E_jᵀS_i): {f_final} or, 2^{log(f_final, 2)}")

# set q
q = 18433
required_condition = q > 4 * f_final
print(f"q=18433 enough: {required_condition}")
if not required_condition:
    print(f"Necessary q: {4 * f_final}")
    print(f"Requirement: 12289, 18433, 40961")
else:
    print(f"Rest: {q/4 - f_final}")
print(log(tail_probability(diff,q/4-2), 2))
